"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .visual_frontend import VisualFrontend
from .acoustic_frontend import AcousticFrontend
from .conformer_block import ConformerBlock

class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch



class AVNetMultiFusion(nn.Module):

    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
        super(AVNetMultiFusion, self).__init__()
        self.visual_frontend = VisualFrontend()  ## the trainable visual frontend
        self.acoustic_frontend = AcousticFrontend()
        # self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        # self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        # self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = ConformerBlock(dim= dModel)
        self.audioEncoder = ConformerBlock(dim = dModel)

        self.audioGating = nn.LSTM(input_size=dModel, hidden_size=256, num_layers=3, dropout=dropout, bidirectional=True)
        self.videoGating = nn.LSTM(input_size=dModel, hidden_size=256, num_layers=3, dropout=dropout, bidirectional=True)
        self.modFusionConv = nn.Conv2d(dModel, dModel, kernel_size=(4, 5), stride=(1, 1), padding=(0, 2), bias=True)

        self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return


    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:
            # audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            # audioBatch = self.audioConv(audioInputBatch)
            batch_size = audioInputBatch.shape[0]
            # print(audioInputBatch.shape,'audio batch')
            audioInputBatch = audioInputBatch.transpose(0,1)
            audioInputBatch = audioInputBatch.transpose(0, 1)
            audioBatch = self.acoustic_frontend(audioInputBatch)
            audioBatch = self.positionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch)
        else:
            audioBatch = None

        if videoInputBatch is not None:
            batch_size = videoInputBatch.shape[0]
            # print(videoInputBatch.shape,'video batch')
            videoInputBatch = videoInputBatch.transpose(0,1)
            videoInputBatch_intermediate = self.visual_frontend(videoInputBatch.unsqueeze(2))
            videoBatch = self.positionalEncoding(videoInputBatch_intermediate)
            videoBatch = self.videoEncoder(videoBatch)
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            # New Change
            # print(audioBatch.shape,videoBatch.shape,'shapeeeeee')
            audioBatch = audioBatch.transpose(0,1)
            gatedAudio, (_h, _c) = self.audioGating(audioBatch)
            gatedVideo, (_h, _c) = self.videoGating(videoBatch)
            gatedAudio = torch.sigmoid(gatedAudio)*videoBatch
            gatedVideo = torch.sigmoid(gatedVideo)*audioBatch
            
            gatedAudio = gatedAudio.unsqueeze(3)
            gatedVideo = gatedVideo.unsqueeze(3)
            audioBatchExt = audioBatch.unsqueeze(3)
            videoBatchExt = videoBatch.unsqueeze(3)
            jointBatch = torch.cat([audioBatchExt, gatedVideo, videoBatchExt, gatedAudio], dim=3)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2).transpose(2, 3)
            jointBatch = self.modFusionConv(jointBatch)
            jointBatch = jointBatch.squeeze(2)
            
            #jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            #jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            #jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)

            
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        jointBatch = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(jointBatch, dim=2)
        # print(outputBatch.shape,'output batch in model file')
        if outputBatch.shape[1]==batch_size:
            return outputBatch.transpose(0,1)
        else:
            return outputBatch
