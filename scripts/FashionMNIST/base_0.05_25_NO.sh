cmdargs=$1
# aggregation_mode: "FedAVG","median", "rfa", "krum", "flame", "NormBound", "trmean"
# attack_method: "label_flip", "targeted_label_flip", "Fang", "MPAF", "Min-Max", "Min-Sum", "Scaling", "DBA", "untargeted_cos"
export CUDA_VISIBLE_DEVICES='5'
hyperparameters04='[{
    "random_seed" : [4],
    "dataset" : ["fmnist"],
    "models" : [{"ConvNet" : 100}],

    "attack_rate" :  [0],
    "attack_method": ["NO"],
    "participation_rate" : [1],

    "alpha" : [0.05],
    "communication_rounds" : [500],
    "local_epochs" : [1],
    "mali_local_epochs": [5],
    "batch_size" : [32],
    "local_optimizer" : [ ["SGD", {"lr": 0.001}]],
    "aggregation_mode" : ["FedAVG", "median", "flame", "NormBound", "krum", "multi-krum", "rfa"],
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
    "uniformed_att": ["True"],
    "lambda_": [1],
    "beta_": [0.4],
    "adv_lr": [0.005],
    "percentile": [25],
    "server_lr": [1.0]
    }]'


RESULTS_PATH="results/"
DATA_PATH="../data/"
CHECKPOINT_PATH="checkpoints/"

python -u codes/run_agrs.py --hp="$hyperparameters04"  --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs
