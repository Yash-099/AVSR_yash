import torch
import numpy as np
from tqdm import tqdm

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode
import time



def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device, trainParams):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    trainingCER = 0
    trainingWER = 0
    iterr = 0
    iterator = tqdm(trainLoader, leave=False, desc="Train",ncols=75)
    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(iterator):

        #clear GPU memory
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        # print(len(inputBatch),'lennnnnnnnnnnnnnnnn')
        # print(inputBatch[0])
        # print(type(inputBatch[0]))
        # print(inputBatch[0].shape)
        #inputBatch = list(inputBatch)
        
        inputBatch[0] = inputBatch[0].transpose(0,1)
        # inputBatch = (inputBatch[0].transpose(0,1),inputBatch[1].transpose(0,1))
        inputBatch[1] = inputBatch[1].transpose(0,1)
        
        inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        opmode = np.random.choice(["AO", "VO", "AV"],
                                  p=[trainParams["aoProb"], trainParams["voProb"], 1-(trainParams["aoProb"]+trainParams["voProb"])])
        if opmode == "AO":
            inputBatch = (inputBatch[0], None)
        elif opmode == "VO":
            inputBatch = (None, inputBatch[1])
        else:
            pass

        optimizer.zero_grad()
        # model.train()
        # print(inputBatch[0].shape,inputBatch[1].shape,'input batch')
        outputBatch = model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            #outputBatch = outputBatch.transpose(0,1)
            # print(outputBatch.shape,targetBatch.shape)
            # print(inputLenBatch.shape,targetLenBatch.shape)
            outputBatch = outputBatch.transpose(0,1)
            loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch, trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, trainParams["spaceIx"])
        iterr += 1
        iterator.set_postfix({"error" : str((trainingLoss/iterr))[:6]})

    trainingLoss = trainingLoss/len(trainLoader)
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)
    return trainingLoss, trainingCER, trainingWER



def evaluate(model, evalLoader, loss_function, device, evalParams):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0
    iterator = tqdm(evalLoader, leave=False, desc="Eval",ncols=75)
    iterr = 0
    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(iterator):
        #clear GPU memory
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

        inputBatch[0] = inputBatch[0].transpose(0,1)
        inputBatch[1] = inputBatch[1].transpose(0,1)
        
        inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
          
        opmode = np.random.choice(["AO", "VO", "AV"],
                                  p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
        if opmode == "AO":
            inputBatch = (inputBatch[0], None)
        elif opmode == "VO":
            inputBatch = (None, inputBatch[1])
        else:
            pass

        model.eval()
        with torch.no_grad():
            outputBatch = model(inputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                outputBatch = outputBatch.transpose(0,1)
                # print(outputBatch.shape,inputLenBatch.shape,targetLenBatch.shape)
                loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

        evalLoss = evalLoss + loss.item()
        if evalParams["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, evalParams["eosIx"])
        elif evalParams["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch, evalParams["beamSearchParams"],
                                                                    evalParams["spaceIx"], evalParams["eosIx"], evalParams["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, evalParams["spaceIx"])
        iterr += 1
        iterator.set_postfix({"error" : str(evalLoss/iterr)[:6]})

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    return evalLoss, evalCER, evalWER
