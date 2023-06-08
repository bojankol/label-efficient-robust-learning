############################# train simple model multiclass doublece ##################
###
### Code for training the simple multiclass model
### Author: Bojan Kolosnjaji
###############################################################################


import os
import time
import argparse
import torch
import shutil
import numpy as np
from dataloader import load_softwaredefects_data
from models.simple_model_multiclass_doublece import LinModel, train, test, compute_psi
from util import init_v_vector_multiclass, get_active_annotators, save_logs_, save_scores_, save_params, save_baseline
from datetime import date

def main():

    parser = argparse.ArgumentParser(description='Simple Model')

    # Basic Information
    parser.add_argument('--iterations', type=int, default=10, metavar='ITER',
                        help='How often should the experiment run (default: 5)')
    parser.add_argument('--task', type=int, default=4, metavar='TSK',
                        help='Load dataset for Task (default: 1)')
    parser.add_argument('--folds', type=int, default=3, metavar='F',
                        help='k-Fold (default: 3')
    parser.add_argument('--stop-crit', type=float, default=0.4, metavar='STPCRT',
                        help='How many percent of clients should left active (default: 0.4)')
    parser.add_argument('--v-init-type', type=str, default='random', metavar='OPT',
                        help='Select how v is initialized  random, uniform (default: random)')

    # Logging
    parser.add_argument('--plot-title', type=str, default='k-Fold Avg Task', metavar='PT',
                        help='Title for the save_plots (default: k-Fold Avg Task)')
    parser.add_argument('--directory', type=str, default='plots/simple_model', metavar='DIR',
                        help='Simple model (default: plots/simple_model)')

    # Random params
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed for numpy and torch, need to set --rnd-active flag (default: 1)')
    parser.add_argument('--rnd-active', default=False)
    parser.add_argument('--rnd-seed', dest='rnd_active', action='store_true')

    # Hyper params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--optim', type=str, default='SGD', metavar='OPT',
                        help='Select optim SGD or Adam (default: SGD)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--interval-length', type=int, default=5, metavar='N',
                        help='How many epochs should v update before weight update takes place')
    parser.add_argument('--lmd', type=float, default='0.00001', metavar='LMD',
                        help='Lambda, value for threshold and v1 reg (default: 0.0001)')
    parser.add_argument('--soft-threshold-timestep', type=float, default='0.0001', metavar='STS',
                        help='Soft theshold timestep (default: 0.0003)')
    parser.add_argument('--psi-weighting', type=float, default='0.0001', metavar='STS',
                        help='How psi is weighted compared to cross entropy (default: 1.)')

    parser.add_argument('--uniform-labeler-weights', type=int, default=0, metavar='UNIW', help='Are the labeler weights uniform (1) or optimized (0)')

    parser.add_argument('--max-epochs', type=int, default=10000, metavar='MAXEP', help='Maximum number of training epochs')
    parser.add_argument("--percent-gt", type=int, default=5, metavar='GT', help='Percentage of ground truth to use (default: 5%')

    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA AVAILABLE")
    else:
        device = torch.device('cpu')

    if args.rnd_active:
        rnd_seed = args.seed
    else:
        rnd_seed = int(time.time())

    np.random.seed(rnd_seed)
    torch.random.manual_seed(rnd_seed)

    print("Labelers uniform?", args.uniform_labeler_weights)
 
    # Set up folders for logs and save parameters

    filepath = f"logs/simple/multiclass/{date.today().strftime('%d-%m-%Y')}_{args.lmd}_{args.percent_gt}"
    print("File path:", filepath)

    try:
        os.mkdir(filepath)
    except FileExistsError:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

    save_params(filepath, args, rnd_seed)

    for iteration in range(args.iterations):
        folds = load_softwaredefects_data('mmc3.arff', device, fold=args.folds)
        print(len(folds))
        input()

        current_fold = 0
        for samples, test_samples, annotations, test_annotations, ground_truth, test_ground_truth in folds:
            print("Fold", current_fold)
            model = LinModel(samples.size()[1])
            model = model.to(device)

            print(samples.shape)
            print(annotations.shape)
            print(test_annotations.shape)


            input()

            samples_n = samples.size()[0]

            print("Initial number of samples:", samples_n)

            num_chosen = round(samples_n*args.percent_gt/100) # number of test questions

            print("Selected for ground truth", num_chosen)

            indices_chosen = np.random.choice(range(samples_n), size=(num_chosen,))

            samples_chosen = samples[indices_chosen,:]
            ground_truth_chosen = ground_truth[indices_chosen]

            if (args.uniform_labeler_weights==1):
                v = init_v_vector_multiclass(annotations.size()[1], type="uniform")
            else:
                v = init_v_vector_multiclass(annotations.size()[1], type="random")
            v = v.to(device)
            v.requires_grad = True

            if args.optim == "SGD":
                optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            elif args.optim == "Adam":
                optim = torch.optim.Adam(model.parameters(), lr=args.lr)
            else:
                raise Exception("Optimizer not implemented")

            psi = compute_psi(model, samples, annotations, ground_truth, v).to(device) * args.psi_weighting
            lmd = torch.tensor(args.lmd).to(device)
            timestep = torch.tensor(args.soft_threshold_timestep).to(device)

            train_loss_hist = []
            test_loss_hist = []
            train_acc_hist = []
            test_acc_hist = []
            test_predictions=[]
            test_auc_score=[]
            v_hist = []

            output_list = []

            # set stop criterion
            active_annotators = get_active_annotators(v)
            goal_annotators = int(active_annotators * args.stop_crit)

            # epoch counter
            epoch = 0
            last_change = 0
            last_active_clients = 0
            # stop criterion
            while active_annotators > goal_annotators:

                if epoch >= args.max_epochs:
                    break

                v, loss, accuracy = train(epoch, optim, model, samples, samples_chosen, annotations, ground_truth, ground_truth_chosen, v, psi, lmd, timestep, args.interval_length, args.uniform_labeler_weights, device)
                active_annotators = get_active_annotators(v)

#                print(f'epoch: [{"%03d"%epoch}], active_annotators: [{"%04d"%active_annotators}], loss: [{loss}], acc: [{accuracy}%], goal_annotators:[{goal_annotators}], last_change:[{last_change}]')

                if epoch % args.interval_length == 0:
                    train_loss_hist.append(loss)
                    train_acc_hist.append(accuracy)
                    v_hist.append(get_active_annotators(v))

                    test_loss, acc, f1_score, precision, recall, test_predictions, test_auc_score, outputs = test(model,
                                                                       test_samples,
                                                                       test_annotations,
                                                                       test_ground_truth,
                                                                       v,
                                                                       psi,
                                                                       lmd)

                    test_loss_hist.append(test_loss)
                    test_acc_hist.append(acc)
                    output_list.append(outputs)
#                    print(f'train epoch: [{"%03d" % epoch}], acc: [{acc}%], active: [{active_annotators}], goal_annotators: [{goal_annotators}]')


                if last_active_clients == get_active_annotators(v):
                    last_change += 1
                else:
                    last_change = 0

                # for more than 100 epochs no change
                if last_change == 1000:
                    break

                epoch += 1

            # save all logs independently and generate the plots later on
#            save_baseline(test_annotations, test_ground_truth, filepath, iteration, current_fold)

            test_predictions_gt = [ test_p[0] for test_p in test_predictions]
            test_predictions_preds = [test_p[1] for test_p in test_predictions]

            test_ground_truth = test_ground_truth.cpu()
            test_annotations = test_annotations.to(device)
            print("Writing results...")
            #save_baseline(test_annotations, test_ground_truth, filepath, iteration,
            #              current_fold)  # test with majority voting (baseline)
            save_scores_(f1_score, precision, recall, test_auc_score, filepath, iteration,
                         current_fold)  # save model performance scores
            # test_predictions =

            save_logs_(train_loss_hist, test_loss_hist, v_hist, train_acc_hist, test_acc_hist, test_samples.cpu(),
                       test_predictions.cpu(), test_auc_score, output_list, filepath, iteration, current_fold)

            current_fold += 1


if __name__ == "__main__":
    main()
