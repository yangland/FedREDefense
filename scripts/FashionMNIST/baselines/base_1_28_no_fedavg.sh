cmdargs=$1
# aggregation_mode: "FedAVG","median", "NormBound","trmean","krum","flame", "RLR"
# attack_method: "AOP", "UAM"
export CUDA_VISIBLE_DEVICES='2'
hyperparameters04='[{
    "random_seed" : [4],
    "dataset" : ["fmnist"],
    "models" : [{"ConvNet" : 100}],

    "attack_rate" :  [0],
    "attack_method": ["NO"],
    "participation_rate" : [1],

    "alpha" : [1.0],
    "communication_rounds" : [500],
    "local_epochs" : [1],
    "mali_local_epochs": [5],
    "batch_size" : [32],
    "local_optimizer" : [ ["SGD", {"lr": 0.001}]],
    "aggregation_mode" : ["FedAVG"],
    "pretrained" : [null],
    "save_model" : [null],
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
    "uniformed_att": ["True"]
    }]'


RESULTS_PATH="results/"
DATA_PATH="../data/"
CHECKPOINT_PATH="checkpoints/"

python -u codes/run_agrs.py --hp="$hyperparameters04"  --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs
