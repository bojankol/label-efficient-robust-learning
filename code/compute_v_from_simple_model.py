import torch
from models.simple_model import LinModel, train, test, compute_psi
from util import init_v_vector, get_active_annotators


def compute_v(samples,
              test_samples,
              annotations,
              test_annotations,
              ground_truth,
              test_ground_truth,
              device,
              optim,
              lmd,
              stop_crit,
              interval_length,
              soft_threshold_timestep,
              lr,
              momentum):

    model = LinModel(samples.size()[1])
    model = model.to(device)
    v = init_v_vector(annotations.size()[1], type="random")
    v = v.to(device)
    v.requires_grad = True

    if optim == "SGD":
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optim == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not implemented")

    psi = compute_psi(model, samples, annotations, ground_truth, v).to(device)

    lmd = torch.tensor(lmd).to(device)
    timestep = torch.tensor(soft_threshold_timestep).to(device)

    # set stop criterion
    active_annotators = get_active_annotators(v)
    goal_annotators = int(active_annotators * stop_crit)

    # epoch counter
    epoch = 0
    last_change = 0
    last_active_clients = 0
    # stop criterion

    print("COMPUTE V FOR ")

    while active_annotators > goal_annotators:
        v, loss, accuracy = train(epoch, optim, model, samples, annotations, ground_truth, v, psi, lmd, timestep, interval_length, device)
        active_annotators = get_active_annotators(v)

        print(f'epoch: [{"%03d"%epoch}], active_annotators: [{"%04d"%active_annotators}], loss: [{loss}], acc: [{accuracy}%]')

        if epoch % interval_length == 0:
            acc, test_loss = test(model,
                                  test_samples,
                                  test_annotations,
                                  test_ground_truth,
                                  v,
                                  psi,
                                  lmd)

            print(f'train epoch: [{"%03d" % epoch}], acc: [{acc}%]')

        if last_active_clients == get_active_annotators(v):
            last_change += 1
        else:
            last_change = 0

        # for more than 100 epochs no change
        if last_change == 100:
            break

        epoch += 1

    return v