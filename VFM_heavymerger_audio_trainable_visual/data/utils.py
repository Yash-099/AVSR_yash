import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax
from config import args
import math


def prepare_main_input(audioFile, visualFeaturesFile, textFile, noise, reqInpLen, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """

    #reading the target from the target file and converting each character to its corresponding index
    with open(textFile, "r") as f:
        trgt = f.readline().strip()[7:]

    trgt = [charToIx[char] for char in trgt]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)

    #the target length must be less than or equal to 100 characters (restricted space where our model will work)
    if trgtLen > 100:
        print("Target length more than 100 characters. Exiting")
        exit()


    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    sampFreq, inputAudio = wavfile.read(audioFile)

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(np.abs(noise))
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise

    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    #computing STFT and taking only the magnitude of it
    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                 boundary=None, padded=False)
    audInp = np.abs(stftVals)
    audInp = audInp.T


    #loading the visual features
    vidInp = np.load(visualFeaturesFile)
    # print('this is shape of the frame', vidInp.shape)

    if args["FACE_DETECTOR_FRAME_MASKING"]:
        #reading the Face Detected results are stored as numpy arrays
        FDarray = np.load(visualFeaturesFile[:-4]+"_FD.npy")

        #matching vidInp and FDarray size and padding on both sides of required array
        if len(vidInp) >= len(FDarray):
            leftPadding = int(np.floor((len(vidInp) - len(FDarray))/2))
            rightPadding = int(np.ceil((len(vidInp) - len(FDarray))/2))
            FDarray = np.pad(FDarray, (leftPadding,rightPadding), "constant") 
        else:
            leftPadding = int(np.floor((len(FDarray) - len(vidInp))/2))
            rightPadding = int(np.ceil((len(FDarray) - len(vidInp))/2))
            vidInp = np.pad(vidInp,((leftPadding,rightPadding),(0,0)), "constant")

        #Multiplying Face detected array with video features to mask the frames without any face
        FDarray = FDarray.reshape((FDarray.shape[0], 1))
        vidInp = vidInp*FDarray
    
    elif args["SUBNET_FRAME_MASKING"]:
        #read subnet face detection results stored in seprate array
        #SDF -> Subnet Detected Faces
        SDFarray = np.load(visualFeaturesFile[:-4]+"_SDF.npy")
        
        #Multiply Visual Features array with subnet predictions to mask frames
        vidInp = vidInp*SDFarray 

    #padding zero vectors to extend the audio and video length to a least possible integer length such that
    #video length = 4 * audio length
    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)

    
    inp = (audInp,vidInp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def prepare_pretrain_input(audioFile, visualFeaturesFile, targetFile, noise, numWords, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    #reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    #if number of words in target is less than the required number of words, consider the whole target
    if len(words) <= numWords:
        trgtNWord = trgt
        sampFreq, inputAudio = wavfile.read(audioFile)

        vidInp = np.load(visualFeaturesFile)

    else:
        #make a list of all possible sub-sequences with required number of words in the target
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)

        #choose the sub-sequence for target according to a softmax distribution of the lengths
        #this way longer sub-sequences (which are more diverse) are selected more often while
        #the shorter sub-sequences (which appear more frequently) are not entirely missed out
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        #reading the start and end times in the video corresponding to the selected sub-sequence
        startTime = float(lines[4+ix].split(" ")[1])
        endTime = float(lines[4+ix+numWords-1].split(" ")[2])
        #loading the audio
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*startTime):int(sampFreq*endTime)]
        # print(sampFreq, inputAudio.shape,'the input audio shape before stft')
        #loading visual features
        videoFPS = videoParams["videoFPS"]
        vidInp = np.load(visualFeaturesFile)
        vidInp = vidInp[int(np.floor(videoFPS*startTime)):int(np.ceil(videoFPS*endTime))]
    
    # print('this is shape of the frame', vidInp.shape)
    #converting each character in target to its corresponding index
    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)


    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]

    ### pad the audio to get atleast 4 STFT vectors
    # if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
    #     padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
    #     inputAudio = np.pad(inputAudio, padding, "constant")
    # inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(np.abs(noise))
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.abs(np.sum(inputAudio**2)/(gain*np.sum(noise**2))))
        inputAudio = inputAudio + noise

    #normalising the audio to unit power
    if math.isnan((np.sum(inputAudio**2)/len(inputAudio))) or np.sum(inputAudio**2)<0:
        inputAudio = np.zeros(inputAudio.shape)
    else:
        inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    #computing the STFT and taking only the magnitude of it
    # _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                #  boundary=None, padded=False)
    # audInp = np.abs(stftVals)
    # audInp = audInp.T
    audInp = inputAudio

    #padding zero vectors to extend the audio and video length to a least possible integer length such that
    #video length = 4 * audio length
    # if len(audInp)/4 >= len(vidInp):
    #     inpLen = int(np.ceil(len(audInp)/4))
    #     leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
    #     rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
    #     audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
    #     leftPadding = int(np.floor((inpLen - len(vidInp))/2))
    #     rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
    #     vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0),(0,0)), "constant")
    # else:
    #     inpLen = len(vidInp)
    #     leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
    #     rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
    #     audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    inpLen = len(vidInp)
    reqInpLen = req_input_length(trgt)
    # print(reqInpLen,'req inp len')
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0),(0,0)), "constant")
        video_inp_len = vidInp.shape[0]
        if 640*reqInpLen>len(audInp):
            audio_padding_left = int(np.floor ( ((640*reqInpLen - len(audInp))/2) ) )
            audio_padding_right = int(np.ceil ( ((640*reqInpLen - len(audInp))/2) ) )
            audInp = np.pad(audInp, ((audio_padding_left,audio_padding_right)), "constant")
        elif 640*reqInpLen < len(audInp):
            audInp = audInp[:640*reqInpLen]
    else:
        if len(audInp)<640*inpLen:
            audio_padding_left = int(np.floor ( ((640*inpLen - len(audInp))/2) ) )
            audio_padding_right = int(np.ceil ( ((640*inpLen - len(audInp))/2) ) )
            # print(audio_padding_left,audio_padding_right)
            # print(audInp.shape,'sja[e')
            audInp = np.pad(audInp, ((audio_padding_left,audio_padding_right)), "constant")
        else:
            len_to_be_cut = 640*inpLen-len(audInp)
            audInp = audInp[int(len_to_be_cut/2):int(len_to_be_cut/2)+640*inpLen]


    inpLen = len(vidInp)
    # print(vidInp.shape,audInp.shape,'shapes')


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    targetBatch = torch.cat([data[1] for data in dataBatch])
    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    targetLenBatch = torch.stack([data[3] for data in dataBatch])
    return inputBatch, targetBatch, inputLenBatch, targetLenBatch



def req_input_length(trgt):
    """
    Function to calculate the minimum required input length from the target.
    Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
    """
    reqLen = len(trgt)
    lastChar = trgt[0]
    for i in range(1, len(trgt)):
        if trgt[i] != lastChar:
            lastChar = trgt[i]
        else:
            reqLen = reqLen + 1
    return reqLen
