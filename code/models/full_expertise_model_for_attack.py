import torch
import torch.nn.functional as F
import torch.nn as nn
from util import replace_not_available_annotations
from custom_optim import soft_threshold_step
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

class FullExpertiseModelGreyBox(nn.Module):

    def __init__(self, dim, p=0):
        """

        :param dim: number of features
        :param annotators:
        """
        super(FullExpertiseModelGreyBox, self).__init__()

#        for i in range(annotators):
#            self.__setattr__(f"Annotator{i}", nn.Linear(dim, 1))

        self.__setattr__(f"Annotator", nn.Linear(dim,1))

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, v, annotators=1):

#        out = []

#        for i in range(annotators):
        out = self.__getattr__(f"Annotator")(self.dropout(x))
 #       out = torch.cat(tuple(out), 1)
        result = torch.sum(out * v, dim=1)
        # KEEP IN MIND THERE IS NO SIGMOID APPLIED HERE
        return result, out


class ConfNetGreyBox(nn.Module):
    # predict prob dist for each annotator/client
    def __init__(self, dim):

        super(ConfNetGreyBox, self).__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

class FullExpertiseModel(nn.Module):

    def __init__(self, dim, annotators=21, p=0):
        """                                                                                                                                                                                               
                                                                                                                                                                                                    
        :param dim: number of features                                                                                                                                                                    
  
        :param annotators:                                                                                                                                                                                
  
        """
        super(FullExpertiseModel, self).__init__()

        for i in range(annotators):
            self.__setattr__(f"Annotator{i}", nn.Linear(dim, 1))

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, v, annotators=21):

        out = []

        for i in range(annotators):
            out.append(self.__getattr__(f"Annotator{i}")(self.dropout(x)))

        out = torch.cat(tuple(out), 1)
        result = torch.sum(out * v, dim=1)
        # KEEP IN MIND THERE IS NO SIGMOID APPLIED HERE
        
        return result, out
    


class ConfNet(nn.Module):
    # predict prob dist for each annotator/client
    
    def __init__(self, dim, annotators=21):
        

        super(ConfNet, self).__init__()
        self.linear = nn.Linear(dim, annotators)

    def forward(self, x):
        out = self.linear(x)
        return out


    

# samples, v, test_samples
def nn_pred(samples, targets, test_samples, device, epochs=200, lr=0.001):

    model = ConfNet(samples.size()[1])
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        optim.zero_grad()
        output = model(samples)
        loss = F.kl_div(F.log_softmax(output, dim=1), targets)
        loss.backward()
        optim.step()

    result = F.softmax(model(test_samples), dim=1)

    #result = (result >= 0.03).float() * result
    #result = torch.t(result) / torch.sum(result, dim=1)
    #result = torch.t(result)
    return result

def nn_pred_greybox(samples, targets, test_samples, device, epochs=200, lr=0.001):
    model = ConfNetGreyBox(samples.size()[1])
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    
    for i in range(epochs):
        optim.zero_grad()
        output = model(samples)
        loss = F.kl_div(F.log_softmax(output, dim=1), targets.unsqueeze(-1))
        loss.backward()
        optim.step()

    result = F.softmax(model(test_samples), dim=1)

    return result

def nn_pred_extended(samples, targets, test_samples, epochs=30, lr=0.001):
    model = ConfNetGreyBox(samples.size()[1])
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        optim.zero_grad()
        output = model(samples)
        loss = F.kl_div(F.log_softmax(output, dim=1), targets)
        loss.backward()
        optim.step()

    result = F.softmax(model(test_samples), dim=1)
    #result = (result >= 0.05).float() * result
    #result = torch.t(result) / torch.sum(result, dim=1)
    #result = torch.t(result)
    return result.double()


def train_eval(output, v, target):
    vy = torch.sigmoid(output) * v
    predictions = torch.sum(vy, dim=1)
    predictions = (predictions > 0.5).float()

#    print(output)
#    print(predictions)
#    print(target)
#    input()

    return accuracy_score(target.cpu(), predictions.cpu())


def eval(output, v, target):
    vy = torch.sigmoid(output) * v
    predictions = torch.sum(vy, dim=1)

    predictions = (predictions > 0.5).float()
    f1 = f1_score(target.cpu(), predictions.cpu())
    precision = precision_score(target.cpu(), predictions.cpu())
    recall = recall_score(target.cpu(), predictions.cpu())

    return accuracy_score(target.cpu(), predictions.cpu()), f1, precision, recall


def eval_greybox(output, v, target): # v should be an arrat
    vy = torch.sigmoid(output) * v
    predictions = torch.sum(vy, dim=1)

    predictions = (predictions > 0.5).float()
    f1 = f1_score(target.cpu(), predictions.cpu())
    precision = precision_score(target.cpu(), predictions.cpu())
    recall = recall_score(target.cpu(), predictions.cpu())

    return accuracy_score(target.cpu(), predictions.cpu()), f1, precision, recall


def compute_psi(model, samples, annotations, ground_truth, v):
    with torch.no_grad():
        output, output_without_v = model(samples, v)

        predictions = replace_not_available_annotations(annotations, output_without_v)

        bce_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
        mse_loss = F.mse_loss(torch.sigmoid(output_without_v), predictions)

        return bce_loss/mse_loss

def compute_psi_greybox(model, samples, annotations, ground_truth, v):
    with torch.no_grad():
        output, output_without_v = model(samples, v)
        predictions = replace_not_available_annotations(annotations, torch.squeeze(output_without_v))

        bce_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
        mse_loss = F.mse_loss(torch.sigmoid(output_without_v), predictions.unsqueeze(-1))

        return bce_loss/mse_loss


def train_greybox(epoch, optim, model, samples, annotations, ground_truth, v, psi, lmd, timestep, v_timestep, device, p=0.0):

    model.train()
    optim.zero_grad()

#    print(model.__getattr__(f"Annotator").weight)
#    print(model.__getattr__(f"Annotator").weight.grad)

    if v.grad is not None:
        v.grad.zero_()

    output, output_without_v = model(samples, v)

    predictions = replace_not_available_annotations(annotations, torch.squeeze(output_without_v))

    bce_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
    mse_loss = psi * F.mse_loss(torch.sigmoid(output_without_v), predictions.unsqueeze(-1))
    reg_loss = lmd * torch.sum(torch.abs(v))

#    loss = bce_loss + mse_loss #+ reg_loss


#    if epoch % v_timestep == 0:
#        loss.backward()
#        optim.step()
#    else:
        # f(x) + g(x) where g(x) is non differentiable, so we split the loss function in two parts
        # f(x) = bce_loss + mse_loss and g(x) is being solved through iterative soft threshold

    loss = bce_loss + mse_loss
    loss.backward()
    optim.step()
    v = soft_threshold_step(v, timestep, lmd, device, p=p)

    accuracy = train_eval(output_without_v, v, ground_truth)

#    print("BCE loss", bce_loss.data, "MSE loss", mse_loss.data, "Reg loss", reg_loss.data)
#    print("Epoch", epoch, "Loss", loss.data, "Accuracy", accuracy)

    return v, loss.item(), accuracy


def test_greybox(model, samples, annotations, targets, v, psi, lmd):
    model.eval()

    with torch.no_grad():
        output, output_without_v = model(samples, v)

        predictions = replace_not_available_annotations(annotations, torch.squeeze(output_without_v, -1))

        bce_loss = F.binary_cross_entropy_with_logits(output, targets)

        mse_loss = psi * F.mse_loss(torch.sigmoid(output_without_v), predictions.unsqueeze(-1))
        reg_loss = lmd * torch.sum(torch.abs(v))

        loss = bce_loss + mse_loss + reg_loss

        accuracy, f1, precision, recall = eval_greybox(output_without_v, v, targets)
        return loss.item(), accuracy, f1, precision, recall
