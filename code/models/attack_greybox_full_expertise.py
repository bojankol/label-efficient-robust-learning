import os
import time
import argparse
import torch
import numpy as np
from dataloader import load_k_fold
from compute_v_from_simple_model import compute_v
from models.full_expertise_model import FullExpertiseModel, train, test, compute_psi, nn_pred
from util import init_v_matrix, get_active_annotators, init_v_alt_matrix, init_v_uniform_matrix, save_logs_, save_scores_, save_params, save_baseline


def main():

    parser = argparse.ArgumentParser(description='Full Expertise Model')

    # Basic Information
    parser.add_argument('--iterations', type=int, default=1, metavar='ITER',
                        help='how often should the experiment run (default: 5)')
    parser.add_argument('--task', type=int, default=1, metavar='T',
                        help='Load dataset for Task (default: 1)')
    parser.add_argument('--folds', type=int, default=5, metavar='F',
                        help='k-Fold (default: 5)')
    parser.add_argument('--stop-crit', type=float, default=0.4, metavar='STPCRT',
                        help='stop criterion, how many active clients should be left (default: 0.4)')
    parser.add_argument('--v-init-type', type=str, default='random', metavar='OPT',
                        help='Select how v is initialized  random, uniform, pretrained(default: random)')

    # Logging
    parser.add_argument('--plot-title', type=str, default='k-Fold Avg Task', metavar='PT',
                        help='Title for the save_plots (default: k-Fold Avg Task)')
    parser.add_argument('--directory', type=str, default='plots/full_expertise_model', metavar='DIR',
                        help='Simple model (default: plots/full_expertise_model)')

    # Random params
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed for numpy and torch, need to set --rnd-active flag (default: 0)')
    parser.add_argument('--rnd-active', default=False)
    parser.add_argument('--rnd-seed', dest='rnd_active', action='store_true')

    # Hyper params
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--interval-length', type=int, default=25, metavar='N',
                        help='After how many epochs should the weights update (default: 25)')
    parser.add_argument('--optim', type=str, default='Adam', metavar='OPT',
                        help='Select optim SGD or Adam (default: SGD)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lmd', type=float, default='0.0003', metavar='LMD',
                        help='Lambda, value for threshold and v reg (default: 0.003)')
    parser.add_argument('--soft-threshold-timestep', type=float, default='0.001', metavar='WD',
                        help='Soft theshold timestep (default: 0.00001)')
    parser.add_argument('--epochs-second-model', type=int, default=50, metavar='N',
                        help='number of epochs for v prediction model (default: 50)')
    parser.add_argument('--lr-second-model', type=float, default=0.0001, metavar='LR',
                        help='learning rate for v prediction model (default: 0.0001)')
    parser.add_argument('--weight-decay', type=float, default='0.', metavar='WD',
                        help='Weight decay (default: 0)')
    parser.add_argument('--dropout', type=float, default='0.', metavar='DO',
                        help='Dropout for gradient (default: 0)')
    parser.add_argument('--dropout-input-layer', type=float, default='0.', metavar='DO',
                        help='Dropout on input layer (default: 0)')

    # Hyper params for simple model if we're initializing v with a pretrained model, for these parameters to be relevant
    # --v-init-type needs to be set to pretrained
    parser.add_argument('--optim-simple-model', type=str, default='Adam', metavar='OPT',
                        help='Select optim SGD or Adam for the simple model(default: SGD)')
    parser.add_argument('--lmd-simple-model', type=float, default='0.0003', metavar='LMD',
                        help='Lambda for the simple model, value for threshold and v reg (default: 0.001)')
    parser.add_argument('--stop-crit-simple-model', type=float, default=0.9, metavar='STPCRT',
                        help='stop criterion for simple model, how many active clients should be left (default: 0.4)')
    parser.add_argument('--interval-length-simple-model', type=int, default=50, metavar='N',
                        help='After how many epochs should the weights update for simple model(default: 100)')
    parser.add_argument('--soft-threshold-timestep-simple-model', type=float, default='0.001', metavar='WD',
                        help='Soft theshold timestep for simple model (default: 0.001)')
    parser.add_argument('--lr-simple-model', type=float, default=0.0001, metavar='LR',
                        help='learning rate for simple model (default: 0.0001)')
    parser.add_argument('--momentum-simple-model', type=float, default=0.9, metavar='M',
                        help='SGD momentum for simple model (default: 0.9)')
    parser.add_argument('--psi-weighting', type=float, default='1.', metavar='STS',
                        help='How psi is weighted compared to cross entropy (default: 1.)')

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

    # Set up folders for logs and save parameters
    filepath = f"logs/full/{int(time.time())}"
    os.mkdir(filepath)

    save_params(filepath, args, rnd_seed)

    for iteration in range(args.iterations):

        folds = load_k_fold(f"data/UAI14_data/class_data_{args.task}.mat", device, fold=args.folds)

        current_fold = 0
        for samples, test_samples, annotations, test_annotations, ground_truth, test_ground_truth in folds:

            num_annotators=annotations.size()[1]
#            print(np.shape(annotations))
            test_scores_annotators=[]

            for annotator in range(num_annotators): # test the attack for all annotators
                print("Annotator:{0}".format(annotator))
                annotations_gpu = annotations.clone() # make a copy of all annotations
                test_scores=[]
                annotations_cpu = annotations_gpu.cpu() # move to cpu to query
                annotation_indices = np.where(annotations_cpu.numpy()[:, annotator]!=-100)[0] # where the annotations are set
	        random.shuffle(annotation_indices)

                while len(annotation_indices)>0: # while not all annotations are flipped
                    
                    test_loss_list=[]
                    print("Annotation indices:{0}".format(annotation_indices))

                    print(annotations_gpu[:,annotator])

                    lmd = torch.tensor(args.lmd).to(device)
                    timestep = torch.tensor(args.soft_threshold_timestep).to(device)

                    model=LinModel(samples.size()[1])
                    model = model.to(device)

                    model = FullExpertiseModel(samples.size()[1], p=args.dropout_input_layer)
                    model = model.to(device)

                    if args.v_init_type == "random":
                        v = init_v_matrix(annotations, device)
                    elif args.v_init_type == "uniform":
                        v = init_v_uniform_matrix(annotations)
                    elif args.v_init_type == "pretrained":
                        trained_v = compute_v(samples,
                                      test_samples,
                                      annotations,
                                      test_annotations,
                                      ground_truth,
                                      test_ground_truth,
                                      device,
                                      args.optim_simple_model,
                                      args.lmd_simple_model,
                                      args.stop_crit_simple_model,
                                      args.interval_length_simple_model,
                                      args.soft_threshold_timestep_simple_model,
                                      args.lr_simple_model,
                                      args.momentum_simple_model)

                        v = init_v_alt_matrix(annotations, trained_v)
                        print("V is initialized with pretrained v")
                    else:
                        raise Exception("Select a initialization type for v")

                    v.requires_grad = True

                    if args.optim == "SGD":
                        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                    elif args.optim == "Adam":
                        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    else:
                        raise Exception("Optimizer not implemented")
                    
                    psi = compute_psi(model, samples, annotations, ground_truth, v).to(device) * args.psi_weighting
                    lmd = torch.tensor(args.lmd).to(device)
                    timestep = torch.tensor(args.soft_threshold_timestep).to(device)


                    for annotation_ind in annotation_indices: # try all annotations
                        
#                        print("Annotation index:{0}".format(annotation_ind))
                        
                        annotations_cpu = annotations_gpu.cpu() # new annotations cpu
                        
#                        print(annotations_cpu[:,annotator])

#                        print(annotation_ind)                                                                                                                                                             
                        annotations_cpu[:,annotator] = label_flip(annotations_cpu[:,annotator], annotation_ind) # flip label
                        
                        #annotations_cpu = annotations_cpu[:,annotator]
                        print(annotations_cpu[:,annotator])
                        input()
                   
                       
                        annotations_gpu_training = annotations_cpu[:,annotator].to(device) # back to gpu for training                                                                                       
                        # now train model                                                                                                                                                                                           #  FullExpertiseModel(samples.size()[1], p=args.dropout_input_layer)
                        model_attack = LinModelGreyBox(samples.size()[1])
                        model_attack = model_attack.to(device)

                        if args.optim == "SGD": # optimizer                                                                                                                                                 
                            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                            optim_attack = torch.optim.SGD(model_attack.parameters(), lr=args.lr, momentum=args.momentum)
                        elif args.optim == "Adam":
                            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
                            optim_attack = torch.optim.Adam(model_attack.parameters(), lr=args.lr)

                        else:
                            raise Exception("Optimizer not implemented")

                        active_annotators = get_active_annotators(v)
                        goal_annotators = int(active_annotators * args.stop_crit)

                        # epoch counter
                        
                        epoch = 0
                        last_change = 0
                        last_active_clients = 0
                        # stop criterion
                        
                        psi_attack = compute_psi_grey_box(model_attack, samples, annotations_gpu_training, ground_truth).to(device) * args.psi_weighting


                        for epoch in range(3000):

                            #v=v.to(device)                                                                                                                                                                

                            loss, accuracy = train_grey_box(epoch, optim_attack, model_attack, samples, annotations_gpu_training, ground_truth, psi_attack,  device)
                            active_annotators = get_active_annotators(v)

#                            print(f'epoch: [{"%03d"%epoch}], loss: [{loss}], acc: [{accuracy}%]')

                            if epoch % args.interval_length == 0:
                                test_loss, acc, f1_score, precision, recall = test_grey_box(model_attack,test_samples,test_annotations[:,annotator], test_ground_truth, psi_attack)


#                                print(f'train epoch: [{"%03d" % epoch}], acc: [{acc}%]')

                                if last_active_clients == get_active_annotators(v):
                                    last_change += 1
                                else:
                                    last_change = 0

                            # for more than 100 epochs no change
                            
                            if last_change == 100:
                                break

                            epoch += 1

                        test_loss, acc, f1_score, precision, recall = test_grey_box(model_attack,test_samples,test_annotations[:,annotator], test_ground_truth,psi_attack)

#                        print(acc)                                                                                                                                                                        

                        test_loss_list.append(test_loss) # flip tryout ended
                        
#                    print(len(test_loss_list))

                    optimal_point = np.argmax(test_loss_list) # which point incurs highest loss
                    
                    print(optimal_point)
                    annotations_cpu = annotations_gpu.cpu()
                    annotations_cpu[:,annotator] = label_flip(annotations_cpu[:,annotator], annotation_indices[optimal_point]) # flip label
                    
                    annotations_gpu = annotations_cpu.to(device) # annotations with flip back to gpu
                    
#                    print(len(annotation_indices))

                    annotation_indices = np.delete(annotation_indices,optimal_point) # this point is flipped, don't take it into account anymore                                                     

                    # set stop criterion                                                                                                                                                                    
                    active_annotators = get_active_annotators(v)
                    goal_annotators = int(active_annotators * args.stop_crit)

                    # epoch counter                                                                                                                                                                         
                    epoch = 0
                    last_change = 0
                    last_active_clients = 0
                    # stop criterion                                                                                                                                                                        

                    v = init_v_vector(annotations_gpu.size()[1], type="random")
                    v = v.to(device)
                    v.requires_grad = True

                        


                    while active_annotators > goal_annotators:

                        v = v.to(device)

                        if epoch >= 3000:
                            break

                        v, loss, accuracy = train(epoch, optim, model, samples, annotations_gpu, ground_truth, v, psi, lmd, timestep, args.interval_length, device) # train model with flipped label        
                        active_annotators = get_active_annotators(v)

                        if last_active_clients == get_active_annotators(v):
                            last_change += 1
                        else:
                            last_change = 0

                        # for more than 100 epochs no change                                                                                                                                                
                        if last_change == 100:
                            break
                        epoch+=1




                    test_loss, acc, f1_score, precision, recall = test(model,test_samples,test_annotations, test_ground_truth,v, psi, lmd) # test model                                                     

                    print("Train accuracy:{0}, Test accuracy:{1}".format(accuracy,acc))


                    test_scores.append((test_loss, acc, f1_score, precision, recall))

                        
                p = torch.tensor(args.dropout).to(device)
                v, loss, accuracy = train(epoch, optim, model, samples, annotations, ground_truth, v, psi, lmd, timestep, args.interval_length, device, p=p)
                active_annotators = get_active_annotators(v)

                print(f'epoch: [{"%03d"%epoch}], active_annotators: [{"%04d"%active_annotators}], loss: [{loss}], acc: [{accuracy}%]')

                if epoch % args.interval_length == 0:
                    train_loss_hist.append(loss)
                    train_acc_hist.append(accuracy)
                    v_hist.append(active_annotators)

                    # Create a way to split the dataset
                    v_pred = nn_pred(samples,
                                     v,
                                     test_samples,
                                     device,
                                     epochs=args.epochs_second_model,
                                     lr=args.lr_second_model)

                    test_loss, acc, f1_score, precision, recall = test(model,
                                                                       test_samples,
                                                                       test_annotations,
                                                                       test_ground_truth,
                                                                       v_pred,
                                                                       psi,
                                                                       lmd)

                    test_loss_hist.append(test_loss)
                    test_acc_hist.append(acc)

                    print(f'train epoch: [{"%03d" % epoch}], acc: [{acc}%]')

                if last_active_clients == get_active_annotators(v):
                    last_change += 1
                else:
                    last_change = 0

                # for more than 200 epochs no change
                if last_change == 200:
                    break

                epoch += 1

            # save all logs independently and generate the plots later on
            save_baseline(test_annotations, test_ground_truth, filepath, iteration, current_fold)
            save_scores_(f1_score, precision, recall, filepath, iteration, current_fold)
            save_logs_(train_loss_hist, test_loss_hist, v_hist, train_acc_hist, test_acc_hist, filepath, iteration,
                       current_fold)

            current_fold += 1


if __name__ == "__main__":
    main()
