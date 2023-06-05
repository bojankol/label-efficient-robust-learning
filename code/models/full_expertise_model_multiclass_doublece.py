import torch
import torch.nn.functional as F
import torch.nn as nn
from util import replace_not_available_annotations
from custom_optim import soft_threshold_step
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

class FullExpertiseModel(nn.Module):

    def __init__(self, dim, annotators=10, p=0):
        """

        :param dim: number of features
        :param annotators:
        """
        super(FullExpertiseModel, self).__init__()

        for i in range(annotators):
            self.__setattr__(f"Annotator{i}", nn.Linear(dim, 10))

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, v, annotators=10):

        out = []

        for i in range(annotators):
            out.append(self.__getattr__(f"Annotator{i}")(self.dropout(x)))
        out = torch.stack(tuple(out), 1)
        softmax_fn = torch.nn.Softmax(dim=2)
        out_result = softmax_fn(out)
        v_repeat = v.view(v.shape[0], v.shape[1], 1).repeat(1,1,v.shape[1])
        result = out_result * v_repeat
        result= torch.sum(result,dim=1)
        # KEEP IN MIND THERE IS NO SIGMOID APPLIED HERE
        return result, out


class ConfNet(nn.Module):
    # predict prob dist for each annotator/client
    def __init__(self, dim, annotators=10):

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


def nn_pred_extended(samples, targets, test_samples, epochs=30, lr=0.001):

    model = ConfNet(samples.size()[1])
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
    softmax_fn = torch.nn.Softmax(dim=2)
    v_repeat = v.view(v.shape[0], v.shape[1], 1).repeat(1,1,v.shape[1])

    vy = softmax_fn(output) * v_repeat

#    vy = torch.sigmoid(output) * v
    predictions = torch.sum(vy, dim=1)


    target = torch.argmax(target, dim=1)
    predictions = torch.argmax(predictions, dim=1)

#    print("v", v_repeat)

#    print("softmax", softmax_fn(output))

#    print("vy", vy)

#    print("Predictions:", predictions)

#    print("Target:", target)


    return accuracy_score(target.cpu(), predictions.cpu())


def eval(output, v, target):
#    vy = torch.sigmoid(output) * v
#    predictions = torch.sum(vy, dim=1)

#    predictions = (predictions > 0.5).float()

    softmax_fn = torch.nn.Softmax(dim=2)
    v_repeat = v.view(v.shape[0], v.shape[1], 1).repeat(1,1,v.shape[1])

    vy = softmax_fn(output) * v_repeat

#    vy = torch.sigmoid(output) * v                                                                                                                                                                         
    predictions = torch.sum(vy, dim=1)

    softmax_fn_1 = torch.nn.Softmax(dim=1)
    predictions = softmax_fn_1(predictions) # make it sum to one by client weight  

   
    auc_score = roc_auc_score(target.cpu(), predictions.cpu(), average='macro', multi_class='ovr')
 
    target = torch.argmax(target, dim=1)
    prediction_labels = torch.argmax(predictions, dim=1)
    
    f1 = f1_score(target.cpu(), prediction_labels.cpu(), average='macro')
    precision = precision_score(target.cpu(), prediction_labels.cpu(), average='macro')
    recall = recall_score(target.cpu(), prediction_labels.cpu(), average='macro')

#    print(predictions, target)

    return accuracy_score(target.cpu(), prediction_labels.cpu()), f1, precision, recall, predictions, auc_score


def compute_psi(model, samples, annotations, ground_truth, v):
    with torch.no_grad():
        output, output_without_v = model(samples, v)

        predictions = replace_not_available_annotations(annotations, output_without_v)


        loss = torch.nn.CrossEntropyLoss()
        ground_truth=torch.argmax(ground_truth, dim=1)
        bce_loss = loss(output, ground_truth)
        mse_loss = F.mse_loss(torch.nn.Softmax(dim=2)(output_without_v), annotations)

        return bce_loss/mse_loss


def train(epoch, optim, model, samples, samples_chosen, indices_chosen, annotations, ground_truth, ground_truth_chosen, v, psi, lmd, timestep, v_timestep, constant_v, device, p=0.0):

    model.train()
    optim.zero_grad()

    if v.grad is not None:
        v.grad.zero_()

    output_chosen, output_without_v_chosen = model(samples_chosen, v[indices_chosen, :])

    output, output_without_v = model(samples, v)

#    print("Model", model.parameters())

#    for param in model.parameters():
#        print(param)

    predictions = replace_not_available_annotations(annotations, output_without_v)

    bce_loss = F.binary_cross_entropy_with_logits(output_chosen, ground_truth_chosen)

    bce_losses_annotation = psi * F.binary_cross_entropy_with_logits(output_without_v, predictions) # mean over all annotators, all data points

    #mse_loss = psi * F.mse_loss(torch.sigmoid(output_without_v), predictions)
    reg_loss = lmd * torch.sum(torch.abs(v))

#   if epoch == 0:
#        print(f"BCE Loss: {bce_loss.item()}, MSE Loss: {mse_loss.item()}")

    loss = bce_loss + bce_losses_annotation + reg_loss

    if epoch % v_timestep == 0:
        loss.backward()
        optim.step()
    else:
        # f(x) + g(x) where g(x) is non differentiable, so we split the loss function in two parts
        # f(x) = bce_loss + mse_loss and g(x) is being solved through iterative soft threshold

        if (constant_v==0):

            v_loss = bce_loss + bce_losses_annotation
            v_loss.backward()
            v_old = v
            v = soft_threshold_step(v, timestep, lmd, device, p=p)

#        print("V difference",v-v_old)
    accuracy = train_eval(output_without_v, v, ground_truth)

    return v, loss.item(), accuracy


def test(model, samples, annotations, targets, v, psi, lmd):
    model.eval()

    with torch.no_grad():
        output, output_without_v = model(samples, v)

        predictions = replace_not_available_annotations(annotations, output_without_v)

        bce_loss = F.binary_cross_entropy_with_logits(output, targets)
        mse_loss = psi * F.mse_loss(torch.sigmoid(output_without_v), predictions)
        reg_loss = lmd * torch.sum(torch.abs(v))

        loss = bce_loss + mse_loss + reg_loss

        accuracy, f1, precision, recall, predictions, auc_score = eval(output_without_v, v, targets)
        
        return loss.item(), accuracy, f1, precision, recall, predictions, auc_score, output
