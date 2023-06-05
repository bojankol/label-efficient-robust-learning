import torch
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
from models.baseline_model import eval_majority, eval_average


def normalize_matrix(mat):
    torch.t(mat) / torch.sum(mat, dim=1)


def init_v_vector(clients, type="random"):
    # nSamples x mAnnotators
    if type == "random":
        v = torch.rand(clients)
    elif type == "uniform":
        v = torch.ones(clients)
    else:
        raise Exception("Unknown type for init_v_vector")

    v = torch.abs(v)
    v = v / torch.sum(v)
    return v

def init_v_vector_multiclass(clients, type="random"):
    # nSamples x mAnnotators
    if type == "random":
        v = torch.rand(clients)
    elif type == "uniform":
        v = torch.ones(clients)
    else:
        raise Exception("Unknown type for init_v_vector")

    v = torch.abs(v)
    v = v / torch.sum(v)
    return v



def init_v_matrix(annotations, device):
    # nSamples x mAnnotators
#    print(np.shape(annotations))
    v = torch.rand(annotations.size()).to(device)
    v = v * (annotations != -100).float()
    v = torch.t(v) / torch.sum(v, dim=1)
    return torch.t(v)

def init_v_matrix_multiclass(annotations, device):
    # nSamples x mAnnotators
    v = torch.rand(annotations.size()[0:2]).to(device)
#    v = v * (annotations != -100).float()
    v_sum = torch.sum(v,dim=1)
    v = v / v_sum.view(v_sum.shape[0], 1).repeat(1,v.shape[1])
    return v


def init_v_uniform_matrix(annotations, device, remove_non_available=True):
    v = torch.ones(annotations.size())
    v=v.to(device)
    if remove_non_available:
        v = v * (annotations != -100.).float()

    v = torch.t(v) / torch.sum(v, dim=1)
    return torch.t(v)

def init_v_uniform_matrix_multiclass(annotations, device, remove_non_available=True):
    v = torch.ones(annotations.size()[0:2]).to(device)

#    print(v.shape)
#    input()
#    if remove_non_available:
#        v = v * (annotations != -100.).float()
        
    v_sum = torch.sum(v,dim=1)
    v = v / v_sum.view(v_sum.shape[0], 1).repeat(1,v.shape[1])

    return v


def init_v_alt_matrix(annotations, v_training):
    # nSamples x mAnnotators
    samples = annotations.size()[0]
    annotators = annotations.size()[1]

    tmp = tuple([v_training for i in range(samples)])

    v = torch.cat(tmp, dim=0).view(samples, -1)
    v = v * (annotations != -100.).float()
    v = v.detach()
    #TODO still need to replace this
    for i, sample in enumerate(v):
        #print(torch.sum(sample))
        if torch.sum(sample).item() == 0:
            print("NO ANNOTATORS FOR SAMPLE")
        #    pass
            #v[i, :] = torch.ones(annotators) / annotators

    v = torch.t(v) / torch.sum(v, dim=1)
    return torch.t(v)

def init_v_alt_matrix_multiclass(annotations, v_training):
    # nSamples x mAnnotators
    samples = annotations.size()[0]
    annotators = annotations.size()[1]

    tmp = tuple([v_training for i in range(samples)])

    v = torch.cat(tmp, dim=0).view(samples, -1)
    v = v * (annotations != -100.).float()
    v = v.detach()
    #TODO still need to replace this
    for i, sample in enumerate(v):
        #print(torch.sum(sample))
        if torch.sum(sample).item() == 0:
            print("NO ANNOTATORS FOR SAMPLE")
        #    pass
            #v[i, :] = torch.ones(annotators) / annotators

    v = torch.t(v) / torch.sum(v, dim=1)
    return torch.t(v)


def replace_not_available_annotations(annotations, replacement):
    with torch.no_grad():
        replacement = replacement.squeeze()
        not_available_annotations = (annotations == -100.).float()
        target_annotations = (not_available_annotations * torch.tensor(100.)) + (
                not_available_annotations * torch.sigmoid(replacement)) + annotations

    return target_annotations

def replace_not_available_annotations_multiclass(annotations,replacement):
    with torch.no_grad():
        not_available_annotations = (annotations == -100.).float()
        softmax_fn = torch.nn.Softmax()
#        print(not_available_annotations.shape)
#        print(softmax_fn(replacement).shape)
        target_annotations = (not_available_annotations * torch.tensor(100.)) + (not_available_annotations * softmax_fn(replacement)) + annotations

    return target_annotations
 
def get_active_annotators(v):
    if v.dim() == 1:
        total = v.size()[0]
    else:
        total = v.shape[0] * v.shape[1]
    inactive = torch.sum((v == 0.).float())

    return (total - inactive).int().item()

def get_active_annotators_multiclass(v):
    if v.dim() == 2:
        total = v.size()[0]
    else:
        total = v.shape[0] * v.shape[1]
    inactive = torch.sum((v == 0.).float())

    return (total - inactive).int().item()


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.double()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps/masked_sums


def create_plot(loss_hist, v_hist, train_acc_hist, test_acc_hist, title="Task 1"):
    warnings.warn("Train and test are changed")

    figure(num=None, figsize=(8, 20), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(3, 1, 1)
    plt.plot(loss_hist)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.subplot(3, 1, 2)
    plt.plot(v_hist)
    plt.xlabel('Iterations')
    plt.ylabel('Active clients')

    plt.subplot(3, 1, 3)
    plt.plot(test_acc_hist, label="test accuracy")
    plt.plot(train_acc_hist, label="train accuracy")
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    return plt


def cut_list(l, length):
    return l[0:length]


def compute_fold_mean(*argv, axis=0):
    result = []
    for arg in argv:
        shortest = min(arg, key=lambda elem: len(elem))
        history_same_length = [cut_list(x, len(shortest)) for x in arg]
        result.append(np.mean(np.array(history_same_length), axis=axis))

    return tuple(result)


def compute_variance(*argv, axis=0):
    result = []

    for arg in argv:
        result.append(np.var(arg, axis=axis))

    return tuple(result)


def save_plots(train_loss_mean, test_loss_mean, v_mean, train_acc_mean, test_acc_mean, iteration, args):
    plt.figure(1)
    plt.plot(v_mean)
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Active clients')
    plt.savefig(f"{args.directory}/{args.plot_title}_iteration_{iteration}_task_{args.task}_clients")
    plt.cla()

    plt.figure(2)
    plt.plot(train_loss_mean, label="Train")
    plt.plot(test_loss_mean, label="Test")
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.directory}/{args.plot_title}_iteration_{iteration}_task_{args.task}_loss")
    plt.cla()

    plt.figure(3)
    plt.plot(train_acc_mean, label="Train accuracy")
    plt.plot(test_acc_mean, label="Test accuracy")
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{args.directory}/{args.plot_title}_iteration_{iteration}_task_{args.task}_accuracy")
    plt.cla()


def save_scores(f1_score, precision, recall, iteration, args):
    f1_mean, precision_mean, recall_mean = compute_fold_mean(f1_score,
                                                             precision,
                                                             recall)

    f1_var, precision_var, recall_var = compute_variance(f1_score,
                                                         precision,
                                                         recall)

    scores = np.array([[f1_mean, f1_var], [precision_mean, precision_var], [recall_mean, recall_var]])
    np.save(f"{args.directory}/{args.plot_title}_iteration_{iteration}_scores", scores)


def generate_base_file(filepath, iteration, current_fold):
    return f"{filepath}/iteration_{iteration}_fold_{current_fold}"


def save_scores_(f1_score, precision, recall, auc_score, filepath, iteration, current_fold):
    base_file = generate_base_file(filepath, iteration, current_fold)

    np.save(f"{base_file}_f1_score", np.array(f1_score))
    np.save(f"{base_file}_precision", np.array(precision))
    np.save(f"{base_file}_recall", np.array(recall))
    np.save(f"{base_file}_aucscore", np.array(auc_score))


def save_baseline(test_annotations, test_ground_truth, filepath, iteration, current_fold):
    base_file = generate_base_file(filepath, iteration, current_fold)

    majority_acc = eval_majority(test_annotations, test_ground_truth)
    average_acc = eval_average(test_annotations, test_ground_truth)

    np.save(f"{base_file}_majority_accuracy", np.array(majority_acc))
    np.save(f"{base_file}_average_accuracy", np.array(average_acc))

def save_logs_(train_loss_hist, test_loss_hist, v_hist, train_acc_hist, test_acc_hist, test_samples, test_predictions, test_auc_score, output_list, filepath, iteration, current_fold):
    base_file = generate_base_file(filepath, iteration, current_fold)

    np.save(f"{base_file}_targets", np.array(test_samples))
    np.save(f"{base_file}_predictions", np.array(test_predictions))
    np.save(f"{base_file}_train_loss", np.array(train_loss_hist))
    np.save(f"{base_file}_test_loss", np.array(test_loss_hist))
    np.save(f"{base_file}_v_loss", np.array(v_hist))
    np.save(f"{base_file}_train_acc", np.array(train_acc_hist))
    np.save(f"{base_file}_test_acc", np.array(test_acc_hist))
    np.save(f"{base_file}_auc_score", np.array(test_auc_score))
    np.save(f"{base_file}_outputs", np.array(output_list))

def save_logs(train_loss_mean, test_loss_mean, v_mean, train_acc_mean, test_acc_mean, iteration, args):
    logs = np.array([train_loss_mean, test_loss_mean, v_mean, train_acc_mean, test_acc_mean])
    np.save(f"{args.directory}/{args.plot_title}_iteration_{iteration}_logs", logs)


def save_params(filepath, args, rnd_seed):
    params = vars(args)
    params["rnd_seed"] = rnd_seed
    json.dump(params, open(f"{filepath}/params.json", 'w'))


def load_params(folder):
    return json.load(open(f"{folder}/params.json"))
