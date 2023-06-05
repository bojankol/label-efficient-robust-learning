import torch
import torch.nn.functional as F
from util import replace_not_available_annotations_multiclass
from custom_optim import soft_threshold_step
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


class LinModel(torch.nn.Module):
    def __init__(self, dim):
        super(LinModel, self).__init__()

        for i in range(10):
            self.__setattr__(f"Linear{i}", torch.nn.Linear(dim, 10))

    def forward(self, x, v_conf):
        out = []
        for i in range(10):
            out.append(self.__getattr__(f"Linear{i}")(x))
#        print("forward")
#        print(out[0].shape)
        out = torch.stack(tuple(out), 1)

#        print(out.shape, v_conf.shape)
        result = torch.sum(out * v_conf, dim=1) ## ??? looks like it's not used
        return result, out

def train_eval(output, v, target):
    softmax_fn = torch.nn.Softmax(dim=2)
#    print(softmax_fn(output).shape, v.shape)
    vy = softmax_fn(output) * v
    predictions = torch.sum(vy, dim=1)
#    predictions = (predictions > 0.5).float()

#    print(target)
#    print(predictions)
    target = torch.argmax(target, dim=1)
    predictions = torch.argmax(predictions, dim=1)

    return accuracy_score(target.cpu(), predictions.cpu())

def eval(output, v, target):
    v_expanded= v.unsqueeze(0).unsqueeze(-1)
    vy = torch.nn.Softmax(dim=2)(output)*v_expanded


    #vy = vy*v
    predictions = torch.sum(vy, dim=1) # sum all the labeler weights

    softmax_fn_1 = torch.nn.Softmax(dim=1)
    predictions = softmax_fn_1(predictions) # make it sum to one by client weight

#    predictions = (predictions > 0.5).float()
    target = torch.argmax(target, dim=1)
    predictions_labels = torch.argmax(predictions, dim=1)

    f1 = f1_score(target.cpu(), predictions_labels.cpu(), average='macro')
    precision = precision_score(target.cpu(), predictions_labels.cpu(), average='macro')
    recall = recall_score(target.cpu(), predictions_labels.cpu(), average='macro')

    targets_binary = torch.nn.functional.one_hot(target)

    auc_score = roc_auc_score(targets_binary.cpu(), predictions.cpu(), average='macro', multi_class='ovr')


    return accuracy_score(target.cpu(), predictions_labels.cpu()), f1, precision, recall, predictions, auc_score


def compute_psi(model, samples, annotations, ground_truth, v):
    with torch.no_grad():
        
        output, output_without_v = model(samples, v)

#        predictions = replace_not_available_annotations_multiclass(annotations, output_without_v)


#        print(output.shape, ground_truth.shape)
        loss = torch.nn.CrossEntropyLoss()
        ground_truth=torch.argmax(ground_truth, dim=1)
        bce_loss = loss(output, ground_truth)
        bce_losses_annotation = F.binary_cross_entropy_with_logits(torch.nn.Softmax(dim=2)(output_without_v),
                                                                         annotations)  # mean over all annotators, all data points

        return bce_loss/bce_losses_annotation


def train(epoch, optim, model, samples, samples_chosen, annotations, ground_truth, ground_truth_chosen, v, psi, lmd, timestep, v_timestep, constant_v, device):

    model.train()
    optim.zero_grad()

    if v.grad is not None:
        v.grad.zero_()

    output_chosen, output_without_v_chosen = model(samples_chosen, v)

    output, output_without_v = model(samples, v)

#    predictions = replace_not_available_annotations_multiclass(annotations, output_without_v)


#    print(output.shape, annotations)
    loss = torch.nn.CrossEntropyLoss()
    ground_truth_decimal = torch.argmax(ground_truth, dim=1)
    bce_loss = loss(output, ground_truth_decimal)
    num_annotators = v.shape[0]

    bce_losses_annotation = psi* F.binary_cross_entropy_with_logits(torch.nn.Softmax(dim=2)(output_without_v), annotations) # mean over all annotators, all data points

    #mse_loss = psi * F.mse_loss(torch.nn.Softmax(dim=2)(output_without_v), annotations)
    reg_loss = lmd * torch.sum(torch.abs(v))

#    if epoch == 0:
#        print(f"BCE Loss: {bce_loss.item()}, MSE Loss: {mse_loss.item()}")

    loss = bce_loss + bce_losses_annotation + reg_loss

    if epoch % v_timestep == 0:
        loss.backward()
        optim.step()
    else:
        # f(x) + g(x) where g(x) is non differentiable, so we split the loss function in two parts
        # f(x) = bce_loss + mse_loss and g(x) is being solved through iterative soft threshold

        if constant_v==0:
            v_loss = bce_loss + bce_losses_annotation
            v_loss.backward()
            v=v.to(device)
            v = soft_threshold_step(v, timestep, lmd, device)

    accuracy = train_eval(output_without_v, v, ground_truth)

    return v, loss.item(), accuracy


def test(model, samples, annotations, targets, v, psi, lmd):
    model.eval()

    with torch.no_grad():
        output, output_without_v = model(samples, v)

#        predictions = replace_not_available_annotations_multiclass(annotations, output_without_v)
#        print(output.shape, ground_truth.shape)

        ground_truth_decimal = torch.argmax(targets, dim=1)
        loss=torch.nn.CrossEntropyLoss()
        bce_loss = loss(output, ground_truth_decimal)

        bce_losses_annotation = psi * F.binary_cross_entropy_with_logits(torch.nn.Softmax(dim=2)(output_without_v),
                                                                         annotations)  # mean over all annotators, all data points

        #mse_loss = psi * F.mse_loss(torch.nn.Softmax(dim=2)(output_without_v), annotations)
        reg_loss = lmd * torch.sum(torch.abs(v))

        loss = bce_loss + bce_losses_annotation + reg_loss

        accuracy, f1, precision, recall, predictions, auc_score = eval(output_without_v, v, targets)
        return loss.item(), accuracy, f1, precision, recall, predictions, auc_score, output
