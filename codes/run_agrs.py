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
from our_attack import *
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
np.set_printoptions(precision=4, suppress=True)
logger = logging.getLogger("logger")
import datetime

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

    train_data_all, test_data = data.get_data(hp["dataset"], args.DATA_PATH)

    # Creating data indices for training and validation splits:
    np.random.seed(hp["random_seed"])
    torch.manual_seed(hp["random_seed"])
    train_data = train_data_all
    client_loaders, test_loader, client_data_subsets = \
        data.get_loaders(train_data, test_data, n_clients=len(model_names),
                         alpha=hp["alpha"], batch_size=hp["batch_size"], 
                         n_data=None, num_workers=4, seed=hp["random_seed"])

    # initialize server and clients
    server = Server(np.unique(model_names), test_loader,
                    num_classes=num_classes, dataset=hp['dataset'])

    initial_model_state = server.models[0].state_dict().copy()
    
    if hp["aggregation_mode"] == "fltrust":
        fltrust_root_dl = get_fltrust_rootds(train_data, sample_siz=100)
    
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
                elif hp["attack_method"] == "UAM":
                    clients.append(Client_UAM(model_name, optimizer_fn, loader,
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
                    
                    
                    if hp["attack_method"] == "UAM":
                        # The first attack action to try
                        x0 = [0.5, 0.5, 1]
                        malicc.search_initial(x0)
                    elif hp["attack_method"] == "untargeted_cos":
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
                mali_client_get_trial_updates(
                    mali_clients, server, hp["local_epochs"], train_type="benign")
            
            print("mali clients benign training - finished")
            
                
            if hp["attack_method"] == "untargeted_cos":
                budget, acc_results0, acc_results1, acc_results2, acc_benign_mean, mali_w, sd_cos = \
                                untargeted_cos_budget_attack(malicc, mali_clients, server, ben_grad_all, 
                                mal_user_grad_ben_mean, model_name, num_classes, xp, hp, K=6, beta_=hp["beta_"], 
                                lambda_=hp["lambda_"], adv_lr = hp["adv_lr"], percentile=hp["percentile"])
                
                for client in mali_clients:
                    client.model.load_state_dict(mali_w)
                
                xp.log({"vali_server": next(iter(acc_results0))[1]})
                xp.log({"acc_benign_mean": next(iter(acc_benign_mean))[1]})
                xp.log({"mali_vali_acc": next(iter(acc_results1))[1]})
                xp.log({"mali_vali_norm_acc": next(iter(acc_results2))[1]})
                xp.log({"attack_budget": budget})
                xp.log({"state_dict_cos": sd_cos})
                
                
        # Both benign and malicous clients compute weight update
        
        for client in participating_clients:
            client.synchronize_with_server(server)
            train_stats = client.compute_weight_update(hp["local_epochs"])

        # def log_krum_selection(mode, multi_k):
        #     selected_clients_ids = server.krum(participating_clients, hp["attack_rate"], multi_k=multi_k)
        #     malicious_count = sum(1 for client in selected_clients_ids if client in mali_ids_all)
        #     mali_select_p = (malicious_count / len(participating_clients))
        #     xp.log({f"{mode}_select_percentage": mali_select_p})


        # # server aggregation
        # if hp["aggregation_mode"] == "FedAVG":
        #     server.fedavg(participating_clients)
        # elif hp["aggregation_mode"] == "ABAVG":
        #     server.abavg(participating_clients)
        # elif hp["aggregation_mode"] == "median":
        #     server.median(participating_clients)
        # elif hp["aggregation_mode"] == "NormBound":
        #     server.normbound(participating_clients,  hp["attack_rate"])
        # elif hp["aggregation_mode"] == "trmean":
        #     server.TrimmedMean(participating_clients, hp["attack_rate"])
        # elif hp["aggregation_mode"] == "krum":
        #     log_krum_selection("single_Krum", multi_k=False)
        # elif hp["aggregation_mode"] == "multi-krum":
        #     log_krum_selection("Multi_Krum", multi_k=True)
        # elif hp["aggregation_mode"] == "RLR":
        #     server.RLR(participating_clients, hp["robustLR_threshold"])
        # elif hp["aggregation_mode"] == "flame":
        #     mali_select_p=server.flame(participating_clients, hp["attack_rate"], hp["wrong_mal"],
        #                  hp["right_ben"], hp["noise"], hp["turn"])
        #     xp.log({"flame_mali_select_precentage": mali_select_p})
        # elif hp["aggregation_mode"] == "foolsgold":
        #     server.foolsgold(participating_clients)
        # elif hp["aggregation_mode"] == "rfa":
        #     server.rfa(participating_clients)
        # elif hp["aggregation_mode"] == "fltrust":
        #     server.fltrust(participating_clients, root_loader=fltrust_root_dl, epochs=hp['local_epochs'])
        # else:
        #     import pdb
        #     pdb.set_trace()
        
        server_lr = hp.get("server_lr", 1)
        mali_select_p = server.server_aggregation(aggregation_mode=hp["aggregation_mode"],
                                  clients=participating_clients,
                                  server_lr = server_lr,
                                  mali_ratio=hp["attack_rate"],
                                  mali_ids_all=mali_ids_all)
        
        xp.log({f"{hp['aggregation_mode']}_select_percentage": mali_select_p})
            
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

            # Save results to Disk
            xp.save_to_disc(path=args.SUBRESULTS_PATH, name=str(exp_id))
            e = int((time.time()-t1)/c_round *
                    (hp['communication_rounds']-c_round))
            print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                  "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))
            logger.info(f"exp total running time: {datetime.timedelta(seconds=(time.time() - t0))}")
    
    # Save model to disk
    server.save_model(path=args.SUBRESULTS_PATH, name=str(exp_id) + ".pt", if_save=hp["save_model"])

    logger.info(f"Saved results to: {args.SUBRESULTS_PATH}/logfile_{exp_id}")
    
    # Delete objects to free up GPU memory
    del server
    clients.clear()
    torch.cuda.empty_cache()




def run():
    experiments_raw = json.loads(args.hp)
    hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(
        x)][args.start:args.end]
    experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

    print("Running {} Experiments..\n".format(len(experiments)))
    for xp_count, xp in enumerate(experiments):
        run_experiment(xp, xp_count, len(experiments), exp_id = xp.hyperparameters["log_id"])


if __name__ == "__main__":
    run()
