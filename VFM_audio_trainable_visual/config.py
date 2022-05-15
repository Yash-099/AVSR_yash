args = dict()


#project structure
args["CODE_DIRECTORY"] = "/home/SharedData/yash/my_files/Nilesh/BTP_files/deep_vfm_yash/VFM_audio_trainable_visual"
args["DATA_DIRECTORY"] = "/home/SharedData/yash/my_files/Nilesh/LRS2_Dataset" #"/home/SharedData/Nilesh/BTP_files/NPTEL_Dataset_v2/Dataset" #"/home/SharedData/Nilesh/LRS2_Dataset"
# args["DATA_DIRECTORY"] = "/home/SharedData/yash/my_files/Nilesh/BTP_files/FD_NPTEL_Dataset/FD_Dataset_v2"
args["DEMO_DIRECTORY"] = "/home/SharedData/yash/my_files/Nilesh/BTP_files/deep_mtl_avsr/VFM_audio_visual/demo/FD_demo"
args["PRETRAINED_MODEL_FILE"] = "/pretrain_weights/37/best_till_now.pt" #"/checkpoints/models/pretrain_001w-step_0000-wer_1.000.pt"#"/final/models/pretrain_045w-step_0280-wer_0.244-v2.pt" #"/final/models/pretrain_061w-step_0260-wer_0.246-v2.pt"
args["TRAINED_MODEL_FILE"] = None #"/final/base_NPTELv2/base_InitLRS2Trained_EncoderFreeze_TrainedonNPTELv2_train-step_0335-wer_0.362.pt" #"/final/models/train-step_0250-wer_0.168-v2.pt" 
args["TRAINED_LM_FILE"] = "/home/SharedData/yash/my_files/Nilesh/BTP_files/deep_mtl_avsr/pretrained/lrs2_language_model.pt"
args["TRAINED_FRONTEND_FILE"] = "/home/SharedData/yash/my_files/Nilesh/BTP_files/deep_mtl_avsr/pretrained/visual_frontend.pt"


#data
args["PRETRAIN_VAL_SPLIT"] = 0.01
args["NUM_WORKERS"] = 2
args["PRETRAIN_NUM_WORDS"] = 45 #1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37, 45, 61
args["MAIN_REQ_INPUT_LENGTH"] = 145
args["CHAR_TO_INDEX"] = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
                         "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18, 
                         "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14, 
                         "X":26, "Z":28, "<EOS>":39}
args["INDEX_TO_CHAR"] = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8", 
                         5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                         11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y", 
                         26:"X", 28:"Z", 39:"<EOS>"}


#audio preprocessing
args["NOISE_PROBABILITY"] = 0#0.25
args["NOISE_SNR_DB"] = 0
args["STFT_WINDOW"] = "hamming"
args["STFT_WIN_LENGTH"] = 0.040
args["STFT_OVERLAP"] = 0.030


#video preprocessing
args["VIDEO_FPS"] = 25
args["ROI_SIZE"] = 112
args["NORMALIZATION_MEAN"] = 0.3645391#0.4161
args["NORMALIZATION_STD"] = 0.1629#0.1688
args["SAVE WEIGHT FILES"] = False
#video postprocessing
args["FACE_DETECTOR_FRAME_MASKING"] = False #Masks the frame of video input which do not have any detected faces using dlib library's face detector results
args["SUBNET_FRAME_MASKING"] = False # Same as face detector based frame masking except here we use a subnetwork's predictions for masking


#training
args["SEED"] = 19220297 #74149355
args["BATCH_SIZE"] = 1
args["STEP_SIZE"] = 16384 #16384 for LRS2 #5657 for NPTEL #10657 for mixEqual #610 for clean_short #2065 for clean_extended #4096 for clean_mixed datasets
args["NUM_STEPS"] = 150
args["SAVE_FREQUENCY"] = 10


#optimizer, scheduler and regularization
args["INIT_LR"] = 0.001
args["FINAL_LR"] = 1e-10
args["LR_SCHEDULER_FACTOR"] = 0.5
args["LR_SCHEDULER_WAIT"] = 4
args["LR_SCHEDULER_THRESH"] = 0.001
args["MOMENTUM1"] = 0.9
args["MOMENTUM2"] = 0.999
args["AUDIO_ONLY_PROBABILITY"] = 0
args["VIDEO_ONLY_PROBABILITY"] = 0
 

#model
args["AUDIO_FEATURE_SIZE"] = 321
args["NUM_CLASSES"] = 40


#transformer architecture
args["PE_MAX_LENGTH"] = 2500
args["TX_NUM_FEATURES"] = 512
args["TX_ATTENTION_HEADS"] = 8
args["TX_NUM_LAYERS"] = 6
args["TX_FEEDFORWARD_DIM"] = 2048
args["TX_DROPOUT"] = 0.1


#beam search
args["BEAM_WIDTH"] = 100
args["LM_WEIGHT_ALPHA"] = 0.5
args["LENGTH_PENALTY_BETA"] = 0.1
args["THRESH_PROBABILITY"] = 0.0001
args["USE_LM"] = True


#testing
args["TEST_DEMO_DECODING"] = "greedy"
args["TEST_DEMO_NOISY"] = False
args["TEST_DEMO_MODE"] = "AV"


if __name__ == '__main__':
    
    for key,value in args.items():
        print(str(key) + " : " + str(value)) 
