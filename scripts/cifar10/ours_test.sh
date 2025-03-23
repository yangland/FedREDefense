cmdargs=$1
# aggregation_mode: "FedAVG","median", "NormBound","trmean","krum","flame", "RLR", "2steps"
# attack_method: "label_flip", "targeted_label_flip", "Fang", "MPAF", "Min-Max", "Min-Sum", "Scaling", "DBA", "untargeted_cos"
# "ConvNet"
export CUDA_VISIBLE_DEVICES='2'
hyperparameters04='[{
    "random_seed" : [4],
    "dataset" : ["cifar10"],
    "models" : [{"resnet18": 20}], 

    "attack_rate" :  [ 0],
    "attack_method": ["NO"],
    "participation_rate" : [1],

    "alpha" : [0.5],
    "communication_rounds" : [500],
    "local_epochs" : [1],
    "mali_local_epochs": [5],
    "batch_size" : [32],
    "local_optimizer" : [ ["SGD", {"lr": 0.001}]],
    "aggregation_mode" : [ "krum"],
    "pretrained" : [null],
    "save_model" : [1],
    "log_frequency" : [1],
    "log_path" : ["new_noniid/"],
    "robustLR_threshold" : [4] ,
    "wrong_mal" : [0],
    "right_ben" : [0],
    "noise" : [0.001],
    "turn" : [0],
    "objective": ["rev_cos"],
    "search_algo": ["MADS"],
    "critical_layer": ["classifier.weight"],
    "sync_mali_mali_train": ["True"],
    "uniformed_att": ["True"],
    "beta_": [0.3],
    "lambda_": [1],
    "adv_lr": [0.001],
    "percentile": [50],
    "server_lr": [1.0],
    "ours_K": [10]
    }]'


RESULTS_PATH="results/"
DATA_PATH="../data/"
CHECKPOINT_PATH="checkpoints/"

python -u codes/run_agrs.py --hp="$hyperparameters04"  --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs
