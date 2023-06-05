import torch
import torch.nn.functional as F
import numpy as np


def soft_threshold_step(v, timestep, threshold, device, p=0.):
    """
    http://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf slide 8 and 9
    """

    #if p > 0:
    #    v.grad = F.dropout(v.grad, p, training=True)
    param_scale = np.linalg.norm(v.cpu().data.numpy().ravel())

    initial_v = v

    v_input = v - (timestep * v.grad)



    v_lt_threshold = (v_input > threshold).float()
    v_st_threshold = (v_input < -threshold).float()

    a = (v_input - threshold) * v_lt_threshold
    b = (v_input + threshold) * v_st_threshold

    v = a + b

    if p > 0.:
        ones = torch.ones(v.size()).to(device)
        x = F.dropout(ones, p, training=True)
        keep_old = (x == 0.).float()
        new_v = (x != 0.).float()
        v = (keep_old * initial_v) + (new_v * v)



    if v.dim() == 1:
        v = v / torch.sum(v)
    else:
        v = torch.t(v) / torch.sum(v, dim=1)
        v = torch.t(v)

    result = v.data.clone()
    result.requires_grad = True



    return result
