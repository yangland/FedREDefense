import os
from torch.utils.data import ConcatDataset
import datetime
import math
from copy import deepcopy
import random
from client import *
from utils import *
from server import Server, MaliCC
from image_synthesizer import Synthesizer
import resource
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
import logging
import shutil
import datetime
from csv_logging import CsvLogging
from our_attack import *
import data_f, models
from search_algo import *
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
np.set_printoptions(precision=4, suppress=True)
logger = logging.getLogger("logger")


channel_dict = {
    "cifar10": 3,
    "cinic10": 3,
    "fmnist": 1,
    "mnist": 1,
}
imsize_dict = {
    "cifar10": (32, 32),
    "cinic10": (32, 32),
    "fmnist": (28, 28),
    "mnist": (28, 28),
}

class_num_dict = {
    "cifar10": 10,
    "cinic10": 10,
    "fmnist" : 10,
    "mnist" : 10,
}

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--hp", default=None, type=str)

parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)

args = parser.parse_args()
curr_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

# args.RESULTS_PATH = os.path.join(args.RESULTS_PATH, str(random.randint(0,1000)))
args.SUBRESULTS_PATH = os.path.join(args.RESULTS_PATH, curr_time)
if not os.path.exists(args.SUBRESULTS_PATH):
    os.makedirs(args.SUBRESULTS_PATH)

master_csv_header = ["exp_num", "exp_id", "dataset", "alpha", "attack_method",  "attack_rate", "agr", "acc"]
master_csv = CsvLogging(f"exp_summary", args.SUBRESULTS_PATH, master_csv_header)

def detection_metric_per_round(real_label, label_pred):
    nobyz = sum(real_label)
    real_label = np.array(real_label)
    label_pred = np.array(label_pred)
    acc = len(label_pred[label_pred == real_label])/label_pred.shape[0]
    recall = np.sum(label_pred[real_label == 1] == 1)/nobyz
    fpr = np.sum(label_pred[real_label == 0] == 1)/(label_pred.shape[0]-nobyz)
    fnr = np.sum(label_pred[real_label == 1] == 0)/nobyz
    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" %
          (acc, recall, fpr, fnr))
    return acc, recall, fpr, fnr, label_pred


def detection_metric_overall_flame(real_label, label_pred):
    nobyz = sum(real_label)
    real_label = np.array(real_label)
    label_pred = np.array(label_pred)
    nosample = label_pred.shape[0]
    fp = np.sum(label_pred[real_label == 0] == 1)
    fn = np.sum(label_pred[real_label == 1] == 0)
    accurate = len(label_pred[label_pred == real_label])
    return accurate, fp, fn, nobyz, nosample


def run_experiment(xp, xp_count, n_experiments, exp_id):
    t0 = time.time()
    
    # Remove existing handlers to prevent duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # print("args.SUBRESULTS_PATH", args.SUBRESULTS_PATH)
    
    logger.addHandler(logging.FileHandler(filename=f'{args.SUBRESULTS_PATH}/log_{xp.hyperparameters["log_id"]}.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running experiment {xp_count+1} of total {n_experiments} \n")
    hp = xp.hyperparameters
    logger.info("Exp parameters:")
    for key, value in hp.items():
        logger.info(f"{key}:{value}")
    num_classes = {"fmnist": 10, "cifar10": 10, "cinic10": 10, "mnist": 10}[hp["dataset"]]

    args.channel = channel_dict[hp['dataset']]
    args.imsize = imsize_dict[hp['dataset']]
    args.dataset = hp['dataset']
    mali_ids_all = []

    logger.info(f"num classes {num_classes}, dsa mode {hp.get('dsa', True)}")
    model_names = [model_name for model_name, k in hp["models"].items()
                   for _ in range(k)]
    
    print("model_names", model_names)
    optimizer, optimizer_hp = getattr(
        torch.optim, hp["local_optimizer"][0]), hp["local_optimizer"][1]

    def optimizer_fn(x): return optimizer(
        x, **{k: hp[k] if k in hp else v for k, v in optimizer_hp.items()})
    
    logger.info(f"dataset : {hp['dataset']}")

    train_data_all, test_data = data_f.get_data(hp["dataset"], args.DATA_PATH)

    # Creating data indices for training and validation splits:
    np.random.seed(hp["random_seed"])
    torch.manual_seed(hp["random_seed"])
    train_data = train_data_all
    client_loaders, test_loader, client_data_subsets = \
        data_f.get_loaders(train_data, test_data, n_clients=len(model_names),
                         alpha=hp["alpha"], batch_size=hp["batch_size"], 
                         n_data=None, num_workers=4, seed=hp["random_seed"])

    # initialize server and clients
    server = Server(np.unique(model_names), optimizer_fn=optimizer_fn, loader=test_loader,
                    num_classes=num_classes, dataset=hp['dataset'])

    initial_model_state = deepcopy(server.models[0].state_dict())
    two_steps = hp.get("two_steps", False)
    # Ensure it's converted to a proper boolean if it's a string
    if isinstance(two_steps, str):
        two_steps = two_steps.lower() == "true"
    
    v_layers_indices = []
    layer_num = -1
    fltrust_root_dl = []
    if hp["aggregation_mode"] == "fltrust":
        fltrust_root_dl = get_fltrust_rootds(train_data, sample_size=100)
    elif two_steps or hp["aggregation_mode"] in ["2steps_flame", "2steps_rfa"]:
        # in 2steps defence, run pre-assessment to get the sensitivity scores
        sensitivity_scores, layer_names, layer_num = server.pre_assessment(model_name = np.unique(model_names)[0],
                                                    num_classes=num_classes, 
                                                    optimizer_fn= optimizer_fn,
                                                    dataset=hp['dataset'], 
                                                    args=args,
                                                    initial_model_state= initial_model_state)
        xp.log({"sensitivity_scores": sensitivity_scores})
        # v_layers_indices = top_k_indices(sensitivity_scores, k=math.floor(layer_num/3))
        # v_layers_indices.sort()
        v_layers_indices = sorted(range(len(sensitivity_scores)), key=lambda i: sensitivity_scores[i], reverse=True)
        xp.log({"v_layers_indices": v_layers_indices})
        logger.info(f"pre-assessment layers order: {', '.join([layer_names[i] for i in v_layers_indices])}")
        
    if hp["attack_method"] == "untargeted_cos":
        # lambda_searcher = OnlineLambdaOptimizer(lambda_init=0.5,
        #                                         tol=1e-5,
        #                                         device=device)
        lambda_searcher = VibrationAwareAttackSampler(alpha=1, beta=1, window_size=50, momentum=0.4, min_step=0.05) 
        
    if hp["attack_rate"] == 0:
        clients = [Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset=hp['dataset'])
                   for i, (loader, model_name) in enumerate(zip(client_loaders, model_names))]
    else:
        clients = []
        for i, (loader, model_name) in enumerate(zip(client_loaders, model_names)):
            if i < (1 - hp["attack_rate"]) * len(client_loaders):
                clients.append(Client(model_name, optimizer_fn, loader,
                               idnum=i, num_classes=num_classes, dataset=hp['dataset']))
            else:
                 # print(i)
                if hp["attack_method"] == "label_flip":
                    clients.append(Client_flip(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                elif hp["attack_method"] == "targeted_label_flip":
                    clients.append(Client_tr_flip(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                elif hp["attack_method"] == "Fang":
                    clients.append(Client_Fang(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                elif hp["attack_method"] == "MPAF":
                    clients.append(Client_MPAF(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                    clients[-1].init_model = initial_model_state
                elif hp["attack_method"] == "Min-Max":
                    clients.append(Client_MinMax(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                elif hp["attack_method"] == "Min-Sum":
                    clients.append(Client_MinSum(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                elif hp["attack_method"] == "Scaling":
                    clients.append(Client_Scaling(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))
                elif hp["attack_method"] == "DBA":
                    clients.append(Client_DBA(model_name, optimizer_fn, loader,
                                   idnum=i, num_classes=num_classes, dataset=hp['dataset']))                  
                elif hp["attack_method"] == "AOP":
                    clients.append(Client_AOP(model_name, optimizer_fn, loader, idnum=i,
                                   num_classes=num_classes, dataset=hp['dataset'], obj=hp['objective']))
                elif hp["attack_method"] == "untargeted_cos":
                    clients.append(Client_UtCos(model_name, optimizer_fn, loader, idnum=i,
                                   num_classes=num_classes, dataset=hp['dataset']))
                else:
                    import pdb
                    pdb.set_trace()
                
                mali_ids_all = list(range(
                        math.ceil((1 - hp["attack_rate"])*len(client_loaders)), len(client_loaders))) 
                logger.info(f"mali client id: {mali_ids_all}")     
                
                if hp["attack_method"] in ["AOP", "UAM", "untargeted_cos"]:
                    # initialize the UAM malicious group's command center

                    pooled_mali_ds = ConcatDataset(
                        [client_data_subsets[i] for i in mali_ids_all])
                    pooled_mali_dl = torch.utils.data.DataLoader(
                        pooled_mali_ds, batch_size=hp["batch_size"], shuffle=True, num_workers=4)
                    
                    malicc = MaliCC(np.unique(model_names)[0], pooled_mali_dl, optimizer_fn, num_classes=num_classes,
                                    dataset=hp['dataset'], mali_ids=mali_ids_all, data = pooled_mali_ds,
                                    search_algo=hp["search_algo"], obj=hp["objective"])
                    
                    
                    if hp["attack_method"] == "untargeted_cos":
                        malicc.sub_loader = malicc.get_sub_dataloader(size=608)


    print(clients[0].model)

    server.number_client_all = len(client_loaders)
    models.print_model(clients[0].model)

    # Start Distributed Training Process
    logger.info("\nStart Distributed Training..\n")
    t1 = time.time()
    xp.log({"prep_time": t1-t0})
    xp.log({"server_val_{}".format(key): value for key,
           value in server.evaluate_ensemble().items()})
    test_accs = []

    logger.info(f"model key {list(server.model_dict.keys())[0]}")

    
    # In each FL communication round
    for c_round in range(1, hp["communication_rounds"]+1):
        logger.info(f"---iter{c_round}/{hp['communication_rounds']}----")
        participating_clients = server.select_clients(
            clients, hp["participation_rate"])
        xp.log({"participating_clients": np.array(
            [c.id for c in participating_clients])})
        # For attack methods that require benign update from clients to construct the malicious upates
        if hp["attack_method"] in ["Fang", "Min-Max", "Min-Sum", "KrumAtt", "UAM", "AOP", "untargeted_cos"] \
            and hp["attack_method"]!="NO" \
            and hp["attack_rate"]!=0:
            # mali clients get benign grads
            mali_clients, mali_ids = get_mali_clients_this_round(
                participating_clients, client_loaders, hp["attack_rate"])
            
            mal_user_grad_ben_mean, mal_user_grad_ben_std, ben_grad_all = \
                mali_clients_get_updates(
                    mali_clients, server, hp["local_epochs"], train_type="benign")
            
            mali_ids.sort()
            print(f"mali clients{mali_ids} benign training - finished")
            
            
            if_PGD = hp.get("if_PGD", True)
            # Ensure it's converted to a proper boolean if it's a string
            if isinstance(if_PGD, str):
                if if_PGD.lower() == "false":
                    if_PGD = False
            
            if hp["attack_method"] == "untargeted_cos":
                budget, server_vali_acc, mali_trained_vali_acc, mali_scaled_vali_acc, mali_normalized_vali_acc, \
                    acc_benign_mean, mali_sd, scaled_cos, trained_cos, mali_benign_grads_cos, normalized_cos = \
                                untargeted_cos_budget_attack(malicc, mali_clients, server, ben_grad_all, 
                                mal_user_grad_ben_mean, model_name, num_classes, xp, hp,
                                K = hp.get("ours_K", 6),
                                beta_ = hp["beta_"], 
                                lambda_ = hp["lambda_"], 
                                adv_lr = hp["adv_lr"], 
                                percentile = hp["percentile"],
                                if_PGD=if_PGD,
                                lambda_searcher = lambda_searcher,
                                search_lambda = False)
                
                # sd_list = decompose_sd(mali_sd, num=len(mali_clients), budget=sd_cos*10)
                # for i in range(len(mali_clients)):
                #     mali_clients[i].mali_sd = sd_list[i]  
                                  
                for client in mali_clients:
                    client.mali_sd = mali_sd
                
                xp.log({"server_vali_acc": next(iter(server_vali_acc))[1]})
                xp.log({"benign_mean_vali_acc": next(iter(acc_benign_mean))[1]})
                xp.log({"mali_trained_vali_acc": next(iter(mali_trained_vali_acc))[1]})
                xp.log({"mali_normalized_vali_acc": next(iter(mali_normalized_vali_acc))[1]})
                xp.log({"mali_scaled_vali_acc": next(iter(mali_scaled_vali_acc))[1]})
                xp.log({"attack_cos_budget": budget})
                xp.log({"trained_cos": trained_cos}) # named parameters
                xp.log({"normalized_cos": normalized_cos}) # state dict
                xp.log({"scaled_cos": scaled_cos}) # state dict
                xp.log({"mali_benign_grads_cos": mali_benign_grads_cos})
                
        # Both benign and malicous clients compute weight update
        
        for client in participating_clients:
            print("client.id", client.id)
            client.synchronize_with_server(server)
            train_stats = client.compute_weight_update(hp["local_epochs"])


        
        server_lr = hp.get("server_lr", 1)
        mali_select_p, selected_clients_ids, clients_weights = server.server_aggregation(aggregation_mode=hp["aggregation_mode"],
                                  clients=participating_clients,
                                  server_lr = server_lr,
                                  mali_ratio=hp["attack_rate"],
                                  mali_ids_all=mali_ids_all,
                                  if_two_steps = two_steps,
                                  v_layers_indices=v_layers_indices,
                                  layer_num = layer_num,
                                  fltrust_root_dl = fltrust_root_dl,
                                  fltrust_epoches= hp['local_epochs'])
        
        xp.log({f"select_percentage": mali_select_p})
        xp.log({"select_ids": {c_round: selected_clients_ids}})
        xp.log({"clients_weights": {c_round: clients_weights}})
            
        if xp.is_log_round(c_round):
            xp.log({'communication_round': c_round,
                   'epochs': c_round*hp['local_epochs']})
            xp.log({key: clients[0].optimizer.__dict__[
                   'param_groups'][0][key] for key in optimizer_hp})
            eval_result = server.evaluate_ensemble().items()
            xp.log({"server_val_{}".format(key): value for key, value in eval_result})
            logger.info({"server_{}_a_{}".format(
                key, hp["alpha"]): value for key, value in eval_result})

            if hp["attack_method"] in ["DBA", "Scaling", "Backdoor", "targeted_label_flip", "UAM", "AOP"]:
                if hp["attack_method"] in ["DBA", "Scaling", "Backdoor"]:
                    att_result = server.evaluate_backdoor_attack().items()
                elif hp["attack_method"] in ["targeted_label_flip"]:
                    att_result = server.evaluate_tr_lf_attack().items()
                elif hp["attack_method"] in ["UAM", "AOP"]:
                    if hp["objective"] == "targeted_label_flip":
                        att_result = server.evaluate_tr_lf_attack().items()
                    elif hp["objective"] in ["label_flip", "rev_cos"]:
                        att_result = server.evaluate_lp_attack(class_num=10).items()
                    elif hp["objective"] == "Backdoor":
                        att_result = server.evaluate_backdoor_attack().items()
                    else:
                        raise Exception("Unknown objective")
                xp.log({"server_att_{}_a_{}".format(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                    key, hp["alpha"]): value for key, value in att_result})
                logger.info({"server_att_{}_a_{}".format(
                    key, hp["alpha"]): value for key, value in att_result})

            xp.log({"epoch_time": (time.time()-t1)/c_round})
            stats = server.evaluate_ensemble()
            test_accs.append(stats['test_accuracy'])

            
            saved_path = xp.save_to_disc(path=args.SUBRESULTS_PATH, id=exp_id)
            e = int((time.time()-t1)/c_round *
                    (hp['communication_rounds']-c_round))
            print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                  "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))
            logger.info(f"exp total running time: {datetime.timedelta(seconds=(time.time() - t0))}")
            logger.info(f"Saved results to: {saved_path}")
    
    # Save model to disk
    server.save_model(path=args.SUBRESULTS_PATH, name=str(exp_id) + ".pt", if_save=hp["save_model"])

    # Delete objects to free up GPU memory
    del server
    clients.clear()
    torch.cuda.empty_cache()
    
    return test_accs




def run():
    experiments_raw = json.loads(args.hp)
    hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(
        x)][args.start:args.end]
    experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

    filename = "master.json"
    # Save to file
    with open(os.path.join(args.SUBRESULTS_PATH, filename), "w") as f:
        json.dump(experiments_raw, f, indent=4)  

    print("Running {} Experiments..\n".format(len(experiments)))
    for xp_count, xp in enumerate(experiments):
        test_accs = run_experiment(xp, xp_count, len(experiments), exp_id = xp.hyperparameters["log_id"])

        hp = xp.hyperparameters
        master_csv.append_save_csv([xp_count,
                                    hp["log_id"],
                                    hp["dataset"],
                                    hp["alpha"],
                                    hp["attack_method"],
                                    hp["attack_rate"], 
                                    hp["aggregation_mode"],
                                    test_accs[-1]
                                    ])


if __name__ == "__main__":
    run()
