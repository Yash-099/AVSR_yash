import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil
import copy

from config import args
from models.av_net import AVNet
from models.av_net_multifusion import AVNetMultiFusion
from data.lrs2_dataset import LRS2Main
from data.lrs2_dataset import FD_NPTEL
from data.utils import collate_fn
from utils.general import num_params, train, evaluate



matplotlib.use("Agg")
np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#declaring the train and validation datasets and their corresponding dataloaders
audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
videoParams = {"videoFPS":args["VIDEO_FPS"]}
noiseParams = {"noiseFile":args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb":args["NOISE_PROBABILITY"], "noiseSNR":args["NOISE_SNR_DB"]}
#trainData = LRS2Main("train", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], audioParams, videoParams, noiseParams)
trainData = FD_NPTEL("train", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], audioParams, videoParams, noiseParams)
trainLoader = DataLoader(trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)
noiseParams = {"noiseFile":args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb":0, "noiseSNR":args["NOISE_SNR_DB"]}
#valData = LRS2Main("val", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], audioParams, videoParams, noiseParams)
valData = FD_NPTEL("val", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], audioParams, videoParams, noiseParams)
valLoader = DataLoader(valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)


#declaring the model, optimizer, scheduler and the loss function
model = AVNetMultiFusion(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
              args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
"""
AVmodel = AVNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
              args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
"""
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"],
                                                 patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"],
                                                 threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
loss_function = nn.CTCLoss(blank=0, zero_infinity=False)


#removing the checkpoints directory if it exists and remaking it
if os.path.exists(args["CODE_DIRECTORY"] + "/checkpoints"):
    while True:
        ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
        if ch == "y":
            break
        elif ch == "n":
            exit()
        else:
            print("Invalid input")
    shutil.rmtree(args["CODE_DIRECTORY"] + "/checkpoints")

os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints")
os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/models")
os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/plots")


#loading the pretrained weights
if args["TRAINED_MODEL_FILE"] is not None:
    print("\n\nPre-Trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
    print("\n\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))
    print("\nLoading the trained model .... \n")
    
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"], map_location=device))
    """
    AVmodel.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"], map_location=device))
          
    AVaudioEncoder_wts = copy.deepcopy(AVmodel.audioEncoder.state_dict())
    AVvideoEncoder_wts = copy.deepcopy(AVmodel.videoEncoder.state_dict())
    
    #model.audioConv.load_state_dict(AVmodel.audioConv.state_dict())
    #model.positionalEncoding.load_state_dict(AVmodel.positionalEncoding.state_dict())
    #model.audioEncoder.load_state_dict(AVaudioEncoder_wts)
    #model.videoEncoder.load_state_dict(AVvideoEncoder_wts)
    model.jointConv.load_state_dict(AVmodel.jointConv.state_dict())
    model.jointDecoder.load_state_dict(AVmodel.jointDecoder.state_dict())
    model.outputConv.load_state_dict(AVmodel.outputConv.state_dict())
    """
    model.to(device)
    print("Loading Done.\n")


# freezing all except modality fusion layers
for param in model.parameters():
    param.requires_grad = False

#unfreezeParams = [model.audioGating.parameters(), model.videoGating.parameters(), model.modFusionConv.parameters()]
unfreezeParams = [model.audioGating.parameters(), model.videoGating.parameters(), model.modFusionConv.parameters(),
                  model.jointConv.parameters(), model.jointDecoder.parameters(), model.outputConv.parameters()] 
for paramlayer in unfreezeParams:
    for param in paramlayer:
      param.requires_grad = True


trainingLossCurve = list()
validationLossCurve = list()
trainingWERCurve = list()
validationWERCurve = list()


#printing the total and trainable parameters in the model
numTotalParams, numTrainableParams = num_params(model)
print("\nNumber of total parameters in the model = %d" %(numTotalParams))
print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))


print("\nTraining the model .... \n")

trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "aoProb":args["AUDIO_ONLY_PROBABILITY"],
               "voProb":args["VIDEO_ONLY_PROBABILITY"]}
valParams = {"decodeScheme":"greedy", "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "aoProb":0, "voProb":0}

minValWER = 1.00

for step in range(args["NUM_STEPS"]):
    
    #clear GPU memeory
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    
    #train the model for one step
    trainingLoss, trainingCER, trainingWER = train(model, trainLoader, optimizer, loss_function, device, trainParams)
    trainingLossCurve.append(trainingLoss)
    trainingWERCurve.append(trainingWER)

    #evaluate the model on validation set
    validationLoss, validationCER, validationWER = evaluate(model, valLoader, loss_function, device, valParams)
    validationLossCurve.append(validationLoss)
    validationWERCurve.append(validationWER)

    #printing the stats after each step
    print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f || Tr.WER: %.3f  Val.WER: %.3f"
          %(step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))

    #make a scheduler step
    scheduler.step(validationWER)


    #saving the model weights and loss/metric curves in the checkpoints directory after every few steps
    if ((validationWER <= minValWER) or (step%args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"]-1)) and (step != 0):
        if (validationWER <= minValWER):
            minValWER = validationWER

        savePath = args["CODE_DIRECTORY"] + "/checkpoints/models/train-step_{:04d}-wer_{:.3f}.pt".format(step, validationWER)
        torch.save(model.state_dict(), savePath)

        plt.figure()
        plt.title("Loss Curves")
        plt.xlabel("Step No.")
        plt.ylabel("Loss value")
        plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/train-step_{:04d}-loss.png".format(step))
        plt.close()

        plt.figure()
        plt.title("WER Curves")
        plt.xlabel("Step No.")
        plt.ylabel("WER")
        plt.plot(list(range(1, len(trainingWERCurve)+1)), trainingWERCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationWERCurve)+1)), validationWERCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/train-step_{:04d}-wer.png".format(step))
        plt.close()


print("\nTraining Done.\n")
