import numpy as np
import torch
import torch.nn.functional as F
from util import replace_not_available_annotations
from custom_optim import soft_threshold_step
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

class LinModel(torch.nn.Module):
    def __init__(self, dim):
        super(LinModel, self).__init__()

        for i in range(21):
            self.__setattr__(f"Linear{i}", torch.nn.Linear(dim, 1))

    def forward(self, x, v_conf):
        out = []
        for i in range(21):
            out.append(self.__getattr__(f"Linear{i}")(x))

        out = torch.cat(tuple(out), 1)

        result = torch.sum(out * v_conf, dim=1)

        return result, out

def train_eval(output, v, target):
    vy = torch.sigmoid(output) * v
    predictions = torch.sum(vy, dim=1)
    predictions = (predictions > 0.5).float()

    return accuracy_score(target.cpu(), predictions.cpu())

def eval(output, v, target):
    auc_score = 0

    vy = torch.sigmoid(output) * v
    predictions_prob = torch.sum(vy, dim=1)

    predictions = (predictions_prob > 0.5).float()
    f1 = f1_score(target.cpu(), predictions.cpu())

    precision = precision_score(target.cpu(), predictions.cpu())
    recall = recall_score(target.cpu(), predictions.cpu())
    try:
        auc_score = roc_auc_score(target.cpu(), predictions_prob.cpu())
    except ValueError:
        print("ValueError: Target:", target.cpu(), "Predictions:", predictions_prob.cpu(), "Output:", output, "V:",v)
        auc_score = 0
    return accuracy_score(target.cpu(), predictions.cpu()), f1, precision, recall, auc_score


def compute_psi(model, samples, annotations, ground_truth, v): # why is it computed like this? scaling?
    with torch.no_grad():
        output, output_without_v = model(samples, v)

        predictions = replace_not_available_annotations(annotations, output_without_v)

        bce_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
        mse_loss = F.mse_loss(torch.sigmoid(output_without_v), predictions)

        return bce_loss/mse_loss


def train(epoch, optim, model, samples, samples_chosen, annotations, ground_truth, ground_truth_chosen, v, psi, lmd, timestep, v_timestep, device, uniform_labeler_weights=0):

    model.train()
    optim.zero_grad()

    if v.grad is not None:
        v.grad.zero_()

    output_chosen, output_without_v_chosen = model(samples_chosen, v)

    output, output_without_v = model(samples, v)

    predictions = replace_not_available_annotations(annotations, output_without_v)

    bce_loss = F.binary_cross_entropy_with_logits(output_chosen, ground_truth_chosen) # loss 1, only use chosen q samples

    num_annotators = v.shape[0]

    bce_losses_annotation = psi* F.binary_cross_entropy_with_logits(output_without_v, predictions) # mean over all annotators, all data points

    #mse_loss = psi * F.mse_loss(torch.sigmoid(output_without_v), predictions) # loss 2
    reg_loss = lmd * torch.sum(torch.abs(v)) # loss 3

#    if epoch == 0:
#        print(f"BCE Loss: {bce_loss.item()}, MSE Loss: {mse_loss.item()}")

    loss = bce_loss + bce_losses_annotation + reg_loss

    if epoch % v_timestep == 0:
        loss.backward()
        optim.step()
    else:
        # f(x) + g(x) where g(x) is non differentiable, so we split the loss function in two parts
        # f(x) = bce_loss + mse_loss and g(x) is being solved through iterative soft threshold

        v_loss = bce_loss + bce_losses_annotation
        v_loss.backward()
#        print(type(v))
        v=v.to(device)
        if int(uniform_labeler_weights)!=1:
            v = soft_threshold_step(v, timestep, lmd, device)

#        print(type(v))
    accuracy = train_eval(output_without_v, v, ground_truth)

    return v, loss.item(), accuracy


def test(model, samples, annotations, targets, v, psi, lmd):
    model.eval()

    with torch.no_grad():
        output, output_without_v = model(samples, v)

        predictions = replace_not_available_annotations(annotations, output_without_v)
        bce_loss = F.binary_cross_entropy_with_logits(output, targets)
        bce_loss_annotation = psi * F.binary_cross_entropy_with_logits(output_without_v, predictions)
        reg_loss = lmd * torch.sum(torch.abs(v))

        loss = bce_loss + bce_loss_annotation + reg_loss

        accuracy, f1, precision, recall, auc_score = eval(output_without_v, v, targets)
        return loss.item(), accuracy, f1, precision, recall, predictions, auc_score, output, bce_loss, bce_loss_annotation, reg_loss
