import tensorflow as tf
import numpy as np
import os
import time
import argparse
import cv2
import json
import sys
#import matplotlib.pyplot as plt
#import scipy.ndimage
#import scipy.io
import csv
from UCI.UCIGeneral import UCIDatasetGeneral
import uci_models as model_generator

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("--architecture", type=str, default='reg', help="architecture")
    parser.add_argument("--bn", type=int, default=0, help="batch normalization")
    parser.add_argument("--bn_full", type=int, default=0, help="batch bn_full")
    parser.add_argument("--norm_start", type=int, default=0, help="norm_start")
    parser.add_argument("--norm_end", type=int, default=0, help="norm_end")
    parser.add_argument("--drop", type=int, default=0, help="dropout")
    parser.add_argument("--dataset_index", type=int, default=7, help="index_of_dataset")
    parser.add_argument("--batchsize", type=int, default=50, help="batchsize")
    parser.add_argument("--epochs", type=int, default=100, help="num_of_epochs")
    parser.add_argument("--depth", type=int, default=8, help="net depth")
    parser.add_argument("--dim", type=int, default=362, help="layer_width")
    parser.add_argument("--lr", type=float, default=0.01, help="learning_rate")
    parser.add_argument("--activation", type=str, default='relu', help="activation_type")
    parser.add_argument("--data_path", type=str, default='/path_to_data/data', help="path_of_all_datasets")
    args = parser.parse_args()

    return args


args = parseArguments()

def identity(x):
    return x

norm_start = args.norm_start
norm_end = args.norm_end
drop = args.drop
batchsize = args.batchsize
depth = args.depth
dim = args.dim
lr = args.lr
epochs = args.epochs
architecture = args.architecture
if args.activation=='id':
    activation = identity
if args.activation=='elu':
    activation = tf.nn.elu
if args.activation=='relu':
    activation = tf.nn.relu
if args.activation=='selu':
    activation = tf.nn.selu
bn = args.bn
bn_full = args.bn_full
data_path = args.data_path
datasets = os.listdir(data_path)
dataset_idx = args.dataset_index

    
sets = ['cardiotocography-3clases','contrac','mushroom','pendigits','magic','chess-krvk','led-display','plant-margin','steel-plates','statlog-vehicle',
        'statlog-heart','waveform','bank','waveform-noise','statlog-shuttle','plant-shape','statlog-landsat','plant-texture','twonorm','connect-4',
        'page-blocks','wall-following','thyroid','oocytes_merluccius_nucleus_4d','hill-valley','semeion','oocytes_merluccius_states_2f','miniboone',
        'ozone','wine-quality-white','letter','chess-krvkp','adult','yeast','cardiotocography-10clases','statlog-image','optical','titanic','ringnorm','abalone'] 
trainset = UCIDatasetGeneral(dataset = datasets[datasets.index(sets[dataset_idx])], root=data_path, train=True, fold = 0)



train_data = trainset.train_data
val_data = trainset.validation_data
test_data = trainset.test_data
num_of_labels = trainset.num_classes
input_dim = trainset.input_dim()

data_node  = tf.placeholder(tf.float32, shape=[None, input_dim], name='data')
labels_node = tf.placeholder(tf.int32, shape=[None], name='labels')
lr_node = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
isTrain_node = tf.Variable(False, name='istrainvar', trainable = False)
tf.add_to_collection('istrainvar',isTrain_node)

train_data = trainset.train_data
val_data = trainset.validation_data
test_data = trainset.test_data

train_labels = trainset.train_labels
val_labels = trainset.validation_labels
test_labels = trainset.test_labels

current_depth = [8,16,32]
#current_depth = [8]
logits = []
loss = []
trainable_vars = []
train_op = []
predictions = []
correct_prediction = []
err = []
optimizer = tf.train.MomentumOptimizer(lr_node, 0.9)
for i in range(len(current_depth)):
    with tf.variable_scope("depth_"+str(current_depth[i])) as scope:
        if architecture=='ml':
            activation = tf.nn.relu
            logits.append(model_generator.model(data_node, input_dim, dim, num_of_labels, current_depth[i], isTrain_node, bn, norm_start, norm_end))
            
        if architecture=='mlorth':
            activation = tf.nn.relu
            logits.append(model_generator.model2(data_node, input_dim, dim, num_of_labels, current_depth[i], isTrain_node, bn, norm_start, norm_end))
        
        if architecture=='reg':
                logits.append(model_generator.reg_architecture(data_node, input_dim, dim, num_of_labels, current_depth[i], isTrain_node, activation, bn, bn_full, norm_start, norm_end))
        if architecture=='rel':
                logits.append(model_generator.rel_architecture(data_node, input_dim, dim, num_of_labels, current_depth[i], isTrain_node, activation, bn, bn_full, norm_start, norm_end))
                
        if architecture=='res':
            logits.append(model_generator.res_architecture(data_node, dim, num_of_labels, activation, current_depth[i], isTrain_node, bn))    
        if architecture=='res2':
            logits.append(model_generator.res_architecture2(data_node, dim, num_of_labels, activation, current_depth[i], isTrain_node, bn))    
                                                
        loss.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits[-1],labels = labels_node)))
        trainable_vars.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"depth_"+str(current_depth[i])))        
        train_op.append(optimizer.minimize(loss[-1], var_list = trainable_vars[-1]))
        
        predictions.append(logits[-1])
        correct_prediction.append(tf.equal(tf.cast(tf.argmax(predictions[-1], 1), tf.int32), labels_node))
        err.append(1 - tf.reduce_mean(tf.cast(correct_prediction[-1], 'float')))

        
        

lr_schedule = [0.1,0.01,0.001]
#lr_schedule = [0.01]
session = tf.Session()
feed_dict = {}
ini = tf.global_variables_initializer()
print("dataset: %s, dataset_index: [%5d], dim: [%5d], activation: %s, bn: [%5d], bn_full: [%5d], norm_start: [%5d], norm_end: [%5d], architecture: %s"  % (str(sets[dataset_idx]), datasets.index(sets[dataset_idx]), dim, activation, bn, bn_full, norm_start, norm_end, architecture))

agg_test_errors = []
agg_smoothed_val_errors = []
for depth_loop in range(len(current_depth)):
    for lr_loop in range(len(lr_schedule)):
        lr = lr_schedule[lr_loop]
        depth = current_depth[depth_loop]
        train_op_net = train_op[depth_loop]
        error = err[depth_loop]
        feed_dict[lr_node] = lr
        test_error_folds = []
        smoothed_val_error_folds = []
    
        for outer_outer in range(1):
            print(outer_outer)
            for outer in range(4):
                trainset = UCIDatasetGeneral(dataset = datasets[datasets.index(sets[dataset_idx])], root=data_path, train=True, fold = outer)
                train_data = trainset.train_data
        
                val_data = trainset.validation_data
                test_data = trainset.test_data
                
                
                train_labels = trainset.train_labels
                val_labels = trainset.validation_labels
                test_labels = trainset.test_labels
                
                session.run(ini)         
                error_plot_train = []
                error_plot_test = []
                error_plot_val = []
     
                for i in range(epochs):
                    if i>50:
                        feed_dict[lr_node] = lr/10
            
                    avg_error = 0
                    perm = np.random.permutation(train_data.shape[0])
                    perm = np.concatenate((perm,perm),0)
                    itter = 0
                    session.run(isTrain_node.assign(True))
                    for idx in range(0,train_data.shape[0],batchsize):
                        batch_idx = perm[idx:idx + batchsize]
                        perm_val = np.random.permutation(test_data.shape[0])
                        feed_dict[data_node] = train_data[batch_idx,:]
                        feed_dict[labels_node] = train_labels[batch_idx]
                        time_end_h5 = time.time()
                        _, error_val = session.run([train_op_net, error], feed_dict = feed_dict)
                        time_end_run    = time.time()
                        avg_error+=error_val
                        itter+=1
                    error_plot_train.append(avg_error/(itter))
                    
                    avg_test_error = 0
                    session.run(isTrain_node.assign(False))
                    for idx in range(test_data.shape[0]):
                        feed_dict[data_node] = test_data[idx:idx+1,:]
                        feed_dict[labels_node] = test_labels[idx:idx+1]
                        error_test = session.run(error, feed_dict = feed_dict)
                        avg_test_error+=error_test
                    error_plot_test.append(avg_test_error/(test_data.shape[0]))
    
                        
                    avg_val_error = 0
                    for idx in range(val_data.shape[0]):
                        feed_dict[data_node] = val_data[idx:idx+1,:]
                        feed_dict[labels_node] = val_labels[idx:idx+1]
                        error_val = session.run(error, feed_dict = feed_dict)
                        avg_val_error+=error_val
                    error_plot_val.append(avg_val_error/(val_data.shape[0]))
                   
                   
#                    print('_______________________________________________________________')
#                    print("dataset: %s, dataset_index: [%5d], dim: [%5d], depth: [%5d], lr: %.8f, activation: %s, bn: [%5d], architecture: %s"  % (str(sets[dataset_idx]), datasets.index(sets[dataset_idx]), dim, current_depth[depth_loop], lr_schedule[lr_loop], activation, bn, architecture))
#                    print("Testing: epoch [%5d], train error: %.8f, test error: %.8f, val_error: %.8f" % (i, error_plot_train[i], error_plot_test[i], error_plot_val[i]))
                
                min_smoothed_val = 1
                for i in range(5,len(error_plot_val)):
                    smoothed_val = np.mean(error_plot_val[i - 5:i])
                    if smoothed_val<min_smoothed_val:
                        min_smoothed_val = smoothed_val
                        test_error = error_plot_test[i - 3]
                test_error_folds.append(test_error)
                smoothed_val_error_folds.append(min_smoothed_val)
                
        agg_test_errors.append(np.mean(test_error_folds))
        agg_smoothed_val_errors.append(np.mean(smoothed_val_error_folds))        
        print("depth: [%5d], lr: %.8f, test_error_mean: %.8f, test_error_std: %.8f, smoothed_val_error_mean: %.8f, smoothed_val_error_std: %.8f"  % (current_depth[depth_loop], lr_schedule[lr_loop], np.mean(test_error_folds), np.std(test_error_folds), np.mean(smoothed_val_error_folds),np.std(smoothed_val_error_folds)))
final_test_error =  agg_test_errors[agg_smoothed_val_errors.index(min(agg_smoothed_val_errors))]       
print("final_test_error: %.8f"  % (final_test_error))