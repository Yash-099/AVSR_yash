import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import args
from models.av_net import AVNet
from models.lrs2_char_lm import LRS2CharLM
from data.lrs2_dataset import LRS2Main
from data.lrs2_dataset import FD_NPTEL
from data.utils import collate_fn
from utils.general import evaluate



np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers":args["NUM_WORKERS"], "pin_memory":True} if gpuAvailable else {}
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if args["TRAINED_MODEL_FILE"] is not None:

    print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))

    #declaring the model,loss function and loading the trained model weights
    model = AVNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                  args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"], map_location=device))
    model.to(device)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=False)


    #declaring the language model
    lm = LRS2CharLM()
    lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
    lm.to(device)
    if not args["USE_LM"]:
        lm = None


    print("\nTesting the trained model .... \n")

    mode = ["AO", "VO", "AV"]
    noise = [False, True]


    for noise in noise:
        
        #declaring the test dataset and test dataloader
        audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
        videoParams = {"videoFPS":args["VIDEO_FPS"]}
        if noise:
            noiseParams = {"noiseFile":args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb":1, "noiseSNR":args["NOISE_SNR_DB"]}
        else:
            noiseParams = {"noiseFile":args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb":0, "noiseSNR":args["NOISE_SNR_DB"]}
        testData = LRS2Main("test", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                            audioParams, videoParams, noiseParams)
        testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)

        beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"],
                            "threshProb":args["THRESH_PROBABILITY"]}

        
        testParams_AO = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
                          "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm, "aoProb":1, "voProb":0}
        if not noise:
            testParams_VO = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
                              "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm, "aoProb":0, "voProb":1}
        
        testParams_AV = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
                          "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm, "aoProb":0, "voProb":0}
        
        #evaluating the model over the test set
        testLoss_AO, testCER_AO, testWER_AO = evaluate(model, testLoader, loss_function, device, testParams_AO)

        #printing test information
        print("\nMODE: %s || DECODING: %s || NOISE: %s" %("AO", args["TEST_DEMO_DECODING"], noise))

        #printing the test set loss, CER and WER
        print("Test Loss: %.6f || Test CER: %.3f || Test WER: %.3f" %(testLoss_AO, testCER_AO, testWER_AO))
        print("\n")

        if not noise:
            #evaluating the model over the test set
            testLoss_VO, testCER_VO, testWER_VO = evaluate(model, testLoader, loss_function, device, testParams_VO)

            #printing test information
            print("\nMODE: %s || DECODING: %s || NOISE: %s" %("VO", args["TEST_DEMO_DECODING"], noise))

            #printing the test set loss, CER and WER
            print("Test Loss: %.6f || Test CER: %.3f || Test WER: %.3f" %(testLoss_VO, testCER_VO, testWER_VO))
            print("\n")


        #evaluating the model over the test set
        testLoss_AV, testCER_AV, testWER_AV = evaluate(model, testLoader, loss_function, device, testParams_AV)

        #printing test information
        print("\nMODE: %s || DECODING: %s || NOISE: %s" %("AV", args["TEST_DEMO_DECODING"], noise))

        #printing the test set loss, CER and WER
        print("Test Loss: %.6f || Test CER: %.3f || Test WER: %.3f" %(testLoss_AV, testCER_AV, testWER_AV))
        print("\n\n")

    print("Testing Done.\n")        

    
else:
    print("Path to the trained model file not specified.\n")
