import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.av_net_with_trainable_frontends import AVNet
from data.lrs2_dataset import LRS2Pretrain
from data.utils import collate_fn
from utils.general import num_params, train, evaluate
from utils.decoders import ctc_greedy_decode, ctc_search_decode
from models.visual_frontend import VisualFrontend

matplotlib.use("Agg")
# np.random.seed(args["SEED"])
# torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
videoParams = {"videoFPS":args["VIDEO_FPS"]}
noiseParams = {"noiseFile":args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb":args["NOISE_PROBABILITY"], "noiseSNR":args["NOISE_SNR_DB"]}
pretrainData = LRS2Pretrain("pretrain", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                            audioParams, videoParams, noiseParams)
pretrainLoader = DataLoader(pretrainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)


model = AVNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
              args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
model.to(device)

visual = VisualFrontend()
visual.load_state_dict(torch.load("../pretrained/visual_frontend.pt"))
visual_dict = visual.state_dict()

whole_model_dict = model.state_dict()

# for k,v in model.state_dict().items():
#     if k[:15]=='visual_frontend':
#         print(v[0])
#         break

pretrained_dict = {'visual_frontend.'+k: v for k, v in visual_dict.items()}
# print(pretrained_dict,'pretrained')
whole_model_dict.update(pretrained_dict)
model.load_state_dict(whole_model_dict)

# for k,v in model.state_dict().items():
#     if k[:15]=='visual_frontend':
#         print(v[0])
#         break


# model = nn.DataParallel(model)
# if args["PRETRAINED_MODEL_FILE"] is not None:
#     print("\n\nPre-trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
#     print("\nLoading the pre-trained model .... \n")
#     model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"], map_location=device))
#     model.to(device)
#     print("Loading Done.\n")

trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "aoProb":args["AUDIO_ONLY_PROBABILITY"],
               "voProb":args["VIDEO_ONLY_PROBABILITY"]}
loss_function = nn.CTCLoss(blank=0, zero_infinity=False)

for name,param in model.named_parameters():
    if name[:15] == 'visual_frontend':
        param.requires_grad=False


numTotalParams, numTrainableParams = num_params(model)
print("\nNumber of total parameters in the model = %d" %(numTotalParams))
print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))


exit()
for i in pretrainLoader:
    a,b,c,d = i
    print(a[0].shape,a[1].shape,'a')
    print(len(b),'b')
    print(c,'c')
    print(d,'d')
    kachra = model((a[0].float().to(device),a[1].float().to(device)))
    print(kachra.shape)
    loss = loss_function(kachra, b, c,d)
    print(loss)
    predictionBatch, predictionLenBatch = ctc_greedy_decode(kachra.detach(), c, trainParams["eosIx"])
    print(kachra.shape)
    print(predictionBatch)
    print(predictionLenBatch)
    break