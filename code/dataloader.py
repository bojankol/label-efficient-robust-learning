import h5py
import pandas as pd
import torch
import numpy as np
from sklearn.utils import shuffle
import math
import pickle
from softwaredefects_gt import read_arff


def covert_data_mturk(filename, file_index):

    f=h5py.File(filename, 'r')
    variables = list(f.items())

    samples = np.transpose(variables[2][1][()])
    annotations = variables[0][1][()]
    ground_truth = np.transpose(variables[1][1][()])

    samples, annotations, ground_truth = shuffle(samples, annotations, ground_truth, random_state=42)

    samples_train = samples[0:200, :]
    annotations_train = annotations[0:200,:]
    ground_truth_train = ground_truth[0:200,:]

    samples_test = samples[200:, :]
    annotations_test = annotations[200:, :]
    ground_truth_test = ground_truth[200:, :]

    print("Training set size: ", samples_train.shape)
    print("Training annotation size:", annotations_train.shape)
    print("Training ground truth size:", ground_truth_train.shape)
    print("Test size:", samples_test.shape )
    num_samples_train = samples_train.shape[0]
    num_samples_test = samples_test.shape[0]

    num_features = samples_train.shape[1]
    num_annotators = annotations_train.shape[0]
    num_annotations_train = np.sum(annotations_train>=0) # only include correct annotations
    annotations_indices = np.where(annotations_train>=0) # only include indices of correct annotations
    print(annotations_indices[0].shape, annotations_indices[1].shape)
    print(max(annotations_indices[0]), max(annotations_indices[1]))
    print("Number of annotations in the training set:", num_annotations_train, "out of ", annotations_train.size)

    # create new input files

    # mturk

    column_list_mturk = ["id", "annotator" ]
    feature_list_mturk = []
    [feature_list_mturk.append("feature_" + str(i)) for i in range(num_features)]
    column_list_mturk.extend(feature_list_mturk)
    column_list_mturk.append("class")
    df_mturk = pd.DataFrame(0, index=np.arange(num_annotations_train), columns=column_list_mturk) # all annotations in one file
    for i in range(num_annotations_train):
        #print(df_mturk.head())
        df_mturk.loc[i, ["id"]]=annotations_indices[0][i] # what image?
        df_mturk.loc[i, ["annotator"]] = annotations_indices[1][i] # what annotator?
        #print(samples[annotations_indices[0][i], :].shape)
        #print(feature_list_mturk)

        #print(df_mturk.loc[i, feature_list_mturk].shape)
        df_mturk.loc[i, feature_list_mturk] = samples_train[annotations_indices[0][i], :]

        #print(annotations[annotations_indices[0][i], annotations_indices[1][i]])
        df_mturk.loc[i, ["class"]] = str(annotations_train[annotations_indices[0][i], annotations_indices[1][i]])

    df_mturk.to_csv("df_mturk_{0}.csv".format(file_index), index=False)

    # gold

    column_list_gold = ["id"]
    feature_list_gold = []
    [feature_list_gold.append("feature_" + str(i)) for i in range(num_features)]
    column_list_gold.extend(feature_list_gold)
    column_list_gold.append("class")
    df_gold = pd.DataFrame(0, index=np.arange(num_samples_train), columns=column_list_gold)
    for i in range(num_samples_train):
        # print(df_mturk.head())
        df_gold.loc[i, ["id"]] = i # what image?
        df_gold.loc[i, feature_list_gold] = samples_train[i, :]
        df_gold.loc[i, ['class']] = str(ground_truth_train[i][0])

    df_gold.to_csv("df_gold_{0}.csv".format(file_index), index=False)

    # test

    column_list_mturk = ["id"]
    feature_list_mturk = []
    [feature_list_mturk.append("feature_" + str(i)) for i in range(num_features)]
    column_list_mturk.extend(feature_list_mturk)
    column_list_mturk.append("class")
    df_test = pd.DataFrame(0, index=np.arange(num_samples_test), columns=column_list_mturk)
    for i in range(num_samples_test):
        # print(df_mturk.head())
        df_test.loc[i, ["id"]] = i # what image?
        df_test.loc[i, feature_list_mturk] = samples_test[i, :]
        df_test.loc[i, ['class']] = str(ground_truth_test[i][0])

    df_test.to_csv("df_test_{0}.csv".format(file_index), index=False)

def load_k_fold(filename, device, fold=5):
    f = h5py.File(filename, 'r')
    variables = list(f.items())

    # [()] https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
    samples = np.transpose(variables[2][1][()])
    annotations = variables[0][1][()]
    ground_truth = np.transpose(variables[1][1][()])

    samples, annotations, ground_truth = shuffle(samples, annotations, ground_truth)
    ground_truth = ground_truth.reshape(-1)

    samples, annotations, ground_truth = torch.from_numpy(samples).float(), torch.from_numpy(annotations).float(), torch.from_numpy(ground_truth).float()
    samples, annotations, ground_truth = samples.to(device), annotations.to(device), ground_truth.to(device)
    samples_chunks = torch.chunk(samples, fold)
    annotations_chunks = torch.chunk(annotations, fold)
    ground_truth_chunks = torch.chunk(ground_truth, fold)

    all_fold = []

    for i in range(fold):

        train_samples = list(samples_chunks)
        del train_samples[i]
        train_samples = torch.cat(tuple(train_samples))

        train_ground_truth = list(ground_truth_chunks)
        del train_ground_truth[i]
        train_ground_truth = torch.cat(tuple(train_ground_truth))

        train_annotations = list(annotations_chunks)
        del train_annotations[i]
        train_annotations = torch.cat(tuple(train_annotations))

        all_fold.append((train_samples, samples_chunks[i], train_annotations, annotations_chunks[i], train_ground_truth, ground_truth_chunks[i]))

    return all_fold


def load_antivirus_data(filename, device, batchsize=100, test_ratio=0.33):

    f = open(filename, 'rb')

    data = pickle.load(f, encoding='latin1')

    # 7409, 20000
    samples = data[0]
    annotations = data[1]
    ground_truth = data[2]

    test_size = math.floor(samples.shape[0] * test_ratio)

    samples, annotations, ground_truth = shuffle(samples, annotations, ground_truth)

    train_samples, test_samples = samples[test_size:, :], samples[:test_size, :]
    train_annotations, test_annotations = annotations[test_size:, :], annotations[:test_size, :]
    train_ground_truth, test_ground_truth = ground_truth[test_size:], ground_truth[:test_size]

    samples = torch.from_numpy(train_samples).long().to(device)
    samples = torch.split(samples, batchsize)
    test_samples = torch.from_numpy(test_samples).long().to(device)

    ground_truth = torch.from_numpy(train_ground_truth).float().to(device)
    ground_truth = torch.split(ground_truth, batchsize)
    test_ground_truth = torch.from_numpy(test_ground_truth).float().to(device)

    annotations = torch.from_numpy(train_annotations).float().to(device)
    annotations = torch.split(annotations, batchsize)
    test_annotations = torch.from_numpy(test_annotations).to(device)

    return samples, test_samples, annotations, test_annotations, ground_truth, test_ground_truth

def label_encoding(input_array, num_classes):
#    print(type(input_array))
    input_shape = list(input_array.shape)
    input_dim = len(input_shape)
    input_shape.append(10)
#    print(input_shape)
    
    output_array = np.zeros(input_shape)
#    print(output_array.shape)
    for i in range(input_shape[0]):
        if input_dim>=2:
            for j in range(input_shape[1]):
                #output_array[i,j,:] = np.zeros(input_shape[2])
#                print(i,j)
#                print(input_array[i,j])
                output_array[i,j,input_array[i,j]]=1
        else:
            output_array[i,input_array[i]]=1
    
    return output_array


def load_softwaredefects_data(file_name, device, fold=5):
    
    samples, annotations, ground_truth = read_arff('data/SoftwareDefects_data/' + file_name)

    annotations=label_encoding(annotations.to_numpy(),10)
    ground_truth=label_encoding(ground_truth.to_numpy(),10)

    samples, annotations, ground_truth = shuffle(samples, annotations, ground_truth)
#    ground_truth = ground_truth.reshape(-1)
    samples = samples.to_numpy()

    samples, annotations, ground_truth = torch.from_numpy(samples).float(), torch.from_numpy(annotations).float(), torch.from_numpy(ground_truth).float()

    samples, annotations, ground_truth = samples.to(device), annotations.to(device), ground_truth.to(device)

    samples_chunks = torch.chunk(samples, fold)
    annotations_chunks = torch.chunk(annotations, fold)
    ground_truth_chunks = torch.chunk(ground_truth, fold)

    all_fold = []

    for i in range(fold):

        train_samples = list(samples_chunks)
        del train_samples[i]
        train_samples = torch.cat(tuple(train_samples))

        train_ground_truth = list(ground_truth_chunks)
        del train_ground_truth[i]
        train_ground_truth = torch.cat(tuple(train_ground_truth))

        train_annotations = list(annotations_chunks)
        del train_annotations[i]
        train_annotations = torch.cat(tuple(train_annotations))

        print(train_samples.shape, train_annotations.shape, train_ground_truth.shape)

        all_fold.append((train_samples, samples_chunks[i], train_annotations, annotations_chunks[i], train_ground_truth, ground_truth_chunks[i]))

    return all_fold



def main():
    for i in range(10):
        print("Load file ",i)
        covert_data_mturk("data/UAI14_data/class_data_{0}.mat".format(i+1), i+1)


if __name__ == "__main__":
    main()
