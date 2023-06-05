import os
import time
import argparse
import torch
import numpy as np
from dataloader import load_k_fold
from models.simple_model_doublece import LinModel, train, test, compute_psi
from util import init_v_vector, get_active_annotators, save_logs_, save_scores_, save_params, save_baseline
import pickle
import shutil
from datetime import date
import random
import copy
import warnings

warnings.filterwarnings("ignore")


def label_flip(t, ind):
    if t[ind]==0:
        t[ind]=1
    else:
        t[ind]=0
    return t


def main():

    parser = argparse.ArgumentParser(description='Simple Model')

    # Basic Information
    parser.add_argument('--iterations', type=int, default=1, metavar='ITER',
                        help='How often should the experiment run (default: 5)')
    parser.add_argument('--task', type=int, default=4, metavar='TSK',
                        help='Load dataset for Task (default: 1)')
    parser.add_argument('--folds', type=int, default=3, metavar='F',
                        help='k-Fold (default: 3')
    parser.add_argument('--stop-crit', type=float, default=0.1, metavar='STPCRT',
                        help='How many percent of clients should left active (default: 0.2)')
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
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--optim', type=str, default='Adam', metavar='OPT',
                        help='Select optim SGD or Adam (default: Adam)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--interval-length', type=int, default=5, metavar='N',
                        help='How many epochs should v update before weight update takes place')
    parser.add_argument('--lmd', type=float, default='0.0001', metavar='LMD',
                        help='Lambda, value for threshold and v1 reg (default: 0.0001)')
    parser.add_argument('--soft-threshold-timestep', type=float, default='0.001', metavar='STS',
                        help='Soft theshold timestep (default: 0.001)')
    parser.add_argument('--psi-weighting', type=float, default='1', metavar='STS',
                        help='How psi is weighted compared to cross entropy (default: 1)')
    parser.add_argument("--percent-gt", type=int, default=1, metavar='GT', help='Percentage of ground truth to use (default: 10%')

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
        rnd_seed = 42

    np.random.seed(rnd_seed)
    torch.random.manual_seed(rnd_seed)

    # Set up folders for logs and save parameters    filepath = f"logs/full/{date.today().strftime('%d-%m-%Y')}_{args.lmd}_{args.task}_{args.percent_gt}"
    filepath = f"logs/attack/simple/{date.today().strftime('%d-%m-%Y')}_{args.lmd}_{args.percent_gt}"
    print("File path:", filepath)

    try:
        os.mkdir(filepath)
    except FileExistsError:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
    save_params(filepath, args, rnd_seed)

    for iteration in range(args.iterations): # multiple experiments?

        folds = load_k_fold(f"data/UAI14_data/class_data_{args.task}.mat", device, fold=args.folds)

        current_fold = 0
        for samples, test_samples, annotations, test_annotations, ground_truth, test_ground_truth in folds: # multiple folds

            num_annotators=annotations.size()[1]
#            print(np.shape(annotations))

            test_scores_annotators=[]

            samples_n = samples.size()[0]

            print("Initial number of samples:", samples_n)

            num_chosen = round(samples_n*args.percent_gt/100) # number of test questions

            print("Selected for ground truth", num_chosen)

            indices_chosen = np.random.choice(range(samples_n), size=(num_chosen,))

            samples_chosen = samples[indices_chosen,:]
            ground_truth_chosen = ground_truth[indices_chosen]

            for annotator in range(num_annotators): # test the attack for all annotators
                print("Annotator:{0}".format(annotator))

                annotations_gpu = annotations.clone() # make a copy of all annotations

                test_scores=[]
                
                annotations_cpu = annotations_gpu.cpu() # move to cpu to query

                
                annotation_indices = np.where(annotations_cpu.numpy()[:, annotator]!=-100)[0] # where the annotations are set

                # execute attack
#                print(annotation_indices)
                random.shuffle(annotation_indices)
#                print("Shuffled")
#                print(annotation_indices)

                while len(annotation_indices)>0: # while not all annotations are flipped
                    test_loss_list=[]
                    test_model_list= []
                    #print("Annotation indices:{0}".format(annotation_indices))

                    #print(annotations_gpu[:,annotator])


                    for annotation_ind in annotation_indices: # try all annotations
                        #print("Annotation index:{0}".format(annotation_ind))
                        annotations_cpu = annotations_gpu.cpu() # new annotations cpu
                        #print("Annotations to flip", annotations_cpu[:,annotator])
#                        print(annotation_ind)
                        annotations_cpu[:,annotator] = label_flip(annotations_cpu[:,annotator], annotation_ind) # flip label

                        #print("Annotations flipped", annotations_cpu[:,annotator])

                        annotations_gpu_train = annotations_cpu.to(device) # back to gpu for training (with the flip)
                    
                        # now train model
                        model = LinModel(samples.size()[1])
                        model = model.to(device)
                        
                        v = init_v_vector(annotations_gpu.size()[1], type="random")
                        v = v.to(device)
                        v.requires_grad = True
                        
                        if args.optim == "SGD": # optimizer
                            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                        elif args.optim == "Adam":
                            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
                        else:
                            raise Exception("Optimizer not implemented")
                        
                        psi = compute_psi(model, samples, annotations_gpu_train, ground_truth, v).to(device) * args.psi_weighting
                        lmd = torch.tensor(args.lmd).to(device)
                        timestep = torch.tensor(args.soft_threshold_timestep).to(device)
                        
                        # set stop criterion
                        active_annotators = get_active_annotators(v)
                        goal_annotators = int(active_annotators * args.stop_crit)

                        # epoch counter
                        epoch = 0
                        last_change = 0
                        last_active_clients = 0
                        # stop criterion
                        while active_annotators > goal_annotators: # training                            

                            if epoch >= 3000:
                                break
                            v=v.to(device)
                            v, loss, accuracy = train(epoch, optim, model, samples, samples_chosen,  annotations_gpu_train, ground_truth, ground_truth_chosen, v, psi, lmd, timestep, args.interval_length, device)
                            active_annotators = get_active_annotators(v)
                        

                            #print(f'epoch: [{"%03d"%epoch}], active_annotators: [{"%04d"%active_annotators}], loss: [{loss}], acc: [{accuracy}%]')

                            if epoch % args.interval_length == 0:
                            
                                test_loss, acc, f1_score, precision, recall,  predictions, auc_score, output, bce_loss, bce_loss_annotation, reg_loss = test(model,test_samples,test_annotations, test_ground_truth,v, psi, lmd)

                                act_annotators = get_active_annotators(v)

                                if epoch % 1000 ==0:
                                    print(f'train epoch: [{"%03d" % epoch}]\ttest_loss: [{"%.3f" % test_loss}]\tacc: [{"%.3f" % acc}%]\tact_annotators:[{"%.3f" % act_annotators}]\t{"%.3f" % bce_loss}\t{"%.3f" % bce_loss_annotation}\t{"%.3f" % reg_loss}]')

                                if last_active_clients == get_active_annotators(v):
                                    last_change += 1
                                else:
                                    last_change = 0

                            # for more than 100 epochs no change
                            if last_change == 100:
                                break

                            epoch += 1
                        test_loss, acc, f1_score, precision, recall,  predictions, auc_score, output, bce_loss, bce_loss_annotation, reg_loss = test(model,test_samples,test_annotations, test_ground_truth,v, psi, lmd)
                        test_model_list.append((copy.deepcopy(model),copy.deepcopy(v)))
                        #print("Ones:", np.where(test_ground_truth==1))
                        #print("Zeros:", np.where(test_ground_truth==0))

                        print("Test loss: {0:.3f}\tacc : {1:.3f}\tf1 score:{2:.3f}\tPR:{3:.3f}\tRC:{4:.3f}\tAUC:{5:.3f}".format(test_loss, acc, f1_score, precision, recall, auc_score))
                        test_loss_list.append(test_loss) # flip tryout ended
                        
                    print("Test loss list length", len(test_loss_list))
                    optimal_point = np.argmax(test_loss_list) # which point incurs highest loss
                    print("Optimal point", optimal_point)
                    optimal_model = test_model_list[optimal_point]
                    annotations_cpu = annotations_gpu.cpu()
                    annotations_cpu[:,annotator] = label_flip(annotations_cpu[:,annotator], annotation_indices[optimal_point]) # flip label with optimal index
                    annotations_gpu = annotations_cpu.to(device) # annotations with flip back to gpu
#                    print(len(annotation_indices))
                    annotation_indices = np.delete(annotation_indices,optimal_point) # this point is flipped, don't take it into account anymore

#                    v, loss, accuracy = train(epoch, optim, model, samples, annotations_gpu, ground_truth, v, psi, lmd, timestep, args.interval_length, device) # train model with flipped label

                    test_loss, acc, f1_score, precision, recall,  predictions, auc_score, output, bce_loss, bce_loss_annotation, reg_loss = test(optimal_model[0],test_samples,test_annotations, test_ground_truth,optimal_model[1], psi, lmd) # test model

                    #print("Annotation: {0}, Train accuracy:{1}, Test accuracy:{2}, Test loss:{3}".format(len(annotation_indices), accuracy,acc, test_loss))

                    test_scores.append((test_loss, acc, f1_score, precision, recall, predictions,auc_score)) # add for this annotation

                
                test_scores_annotators.append(test_scores) # add for this annotator
                pickle.dump(test_scores_annotators, open('attack_simple_test_scores_1_{0}_{1}_{2}_{3}_temp_05062023.bin'.format(current_fold, iteration, args.lmd, args.percent_gt), 'wb+')) # save current results


            current_fold+=1
        
    
        
            test_ground_truth = test_ground_truth.cpu()
            # save all logs independently and generate the plots later on
            save_baseline(test_annotations, test_ground_truth, filepath, iteration, current_fold)
            save_scores_(f1_score, precision, recall, filepath, iteration, current_fold)
 #          save_logs_(train_loss_hist, test_loss_hist, v_hist, train_acc_hist, test_acc_hist, filepath, iteration, current_fold)



if __name__ == "__main__":
    main()