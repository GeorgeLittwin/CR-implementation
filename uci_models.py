import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow.contrib.layers as layers



LAYER_NORM_BIAS_DEFAULT_NAME = "ln_bias"
LAYER_NORM_GAIN_DEFAULT_NAME = "ln_gain"
LAYER_NORMALIZATION_DEFAULT_NAME = "layer_normalization"



def batchnorm(inputT, is_training=False, scope=None):
#    return inputT
    
    # Note: is_training is tf.placeholder(tf.bool) type
    is_training = tf.get_collection('istrainvar')[0]
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope,
                                      reuse=True))
 
 
 
def LN(
        input_pre_nonlinear_activations,
        epsilon=1e-5,
        name=LAYER_NORMALIZATION_DEFAULT_NAME,
):
    """
    Layer normalizes a 2D tensor along its second axis, which corresponds to
    normalizing within a layer.
    :param input_pre_nonlinear_activations:
    :param input_shape:
    :param name: Name for the variables in this layer.
    :param epsilon: The actual normalized value is
    ```
        norm = (x - mean) / sqrt(variance + epsilon)
    ```
    for numerical stability.
    :return: Layer-normalized pre-non-linear activations
    """
    input_shape = input_pre_nonlinear_activations.get_shape()
    mean, variance = tf.nn.moments(input_pre_nonlinear_activations, [1],
                                   keep_dims=True)
    normalised_input = (input_pre_nonlinear_activations - mean) / tf.sqrt(
        variance + epsilon)
    with tf.variable_scope(name):
        gains = tf.get_variable(
            LAYER_NORM_GAIN_DEFAULT_NAME,
            input_shape,
            initializer=tf.constant_initializer(1.),
        )
        biases = tf.get_variable(
            LAYER_NORM_BIAS_DEFAULT_NAME,
            input_shape,
            initializer=tf.constant_initializer(0.),
        )
    return normalised_input * gains + biases

def get_orth_weights2(shape):
    mat = np.random.normal(size = shape)/np.sqrt(shape[0])
    U, S, V = np.linalg.svd(np.matmul(mat,mat.T))
    U = U[0:shape[0],0:shape[1]]
    U = U.astype('float32')
    weights1 = tf.get_variable('weight1s',initializer = U.astype('float32'))
#    weights1 = tf.get_variable('weights1',initializer = np.zeros((shape[1],shape[1])).astype('float32'))
#    weights1 = tf.get_variable("weights1", [shape[1], shape[1]], tf.float32,
#                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))
#    weights2 = tf.get_variable("weights2", [shape[1], shape[1]], tf.float32,
#                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))                            
    biases1 = tf.get_variable('biases1',[shape[-1]], initializer=tf.constant_initializer(0.0))
    
    weights2 = tf.get_variable('weights2',initializer = U.astype('float32'))
#    weights2 = tf.get_variable('weights2',initializer = np.zeros((shape[1],shape[1])).astype('float32'))
#    biases2 = tf.get_variable('biases2',[shape[-1]], initializer=tf.constant_initializer(0.0))
    
    tf.add_to_collection('l2_n',(tf.nn.l2_loss(weights1 - weights2)))
#    tf.add_to_collection('l2_n',(tf.nn.l2_loss(biases1 - biases2)))
    
    tf.add_to_collection('weights',weights1)
    tf.add_to_collection('weights',biases1)
    tf.add_to_collection('weights',weights2)
#    tf.add_to_collection('weights',biases2)
    return weights1, biases1, weights2
    
def get_orth_weights3(shape):
    mat = np.random.normal(size = shape)
    U, S, V = np.linalg.svd(np.matmul(mat,mat.T))
    U = U[0:shape[0],0:shape[1]]
    U = U.astype('float32')
#    weights1 = tf.get_variable('weight1s',initializer = U)
    weights1 = tf.get_variable('weights1',initializer = np.zeros((shape[1],shape[1])).astype('float32'))
#    weights1 = tf.get_variable("weights1", [shape[1], shape[1]], tf.float32,
#                                 tf.random_normal_initializer(stddev=0.01/np.sqrt(shape[1])))
    biases1 = tf.get_variable('biases1',[shape[-1]], initializer=tf.constant_initializer(0.0))
    
    weights2 = tf.get_variable('weights2',initializer = U)
#    weights2 = tf.get_variable('weights2',initializer = np.zeros((shape[1],shape[1])).astype('float32'))
#    biases2 = tf.get_variable('biases2',[shape[-1]], initializer=tf.constant_initializer(0.0))
    
    tf.add_to_collection('l2_n',(tf.nn.l2_loss(weights1 - weights2)))
#    tf.add_to_collection('l2_n',(tf.nn.l2_loss(biases1 - biases2)))
    
    tf.add_to_collection('weights',weights1)
    tf.add_to_collection('weights',biases1)
    tf.add_to_collection('weights',weights2)
#    tf.add_to_collection('weights',biases2)
    return weights1, biases1, weights2
    
def get_orth_weights(shape):
    mat = np.random.normal(size = (max(shape),max(shape)))
    U, S, V = np.linalg.svd(np.matmul(mat,mat.T))
    U = U[0:shape[0],0:shape[1]]
    U = U.astype('float32')
    weights1 = tf.get_variable('weight1s',initializer = U)
    
#    weights1 = tf.get_variable("weights1", [shape[0], shape[1]], tf.float32,
#                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))
                                 
    biases1 = tf.get_variable('biases1',[shape[-1]], initializer=tf.constant_initializer(0.0))
    
#    tf.add_to_collection('weights',weights1)
#    tf.add_to_collection('weights',biases1)

    return weights1, biases1
    
def get_weights(shape, double = False):
    mat = np.random.normal(size = shape)/np.sqrt(shape[1])
    weights1 = tf.get_variable("weights1", initializer = mat.astype('float32'))
                                 
    biases1 = tf.get_variable('biases1',[shape[-1]], initializer=tf.constant_initializer(0.0))
    if double:
        weights2 = tf.get_variable("weights2", initializer = mat.astype('float32'))
        return weights1,weights2,biases1
    else:
        return weights1, biases1


def layer_orth2(inp, shape, isTrainVar, scope, bn = False, activation = False):
    weights1, bias1, weights2 = get_orth_weights2(shape)
    tmp1 = tf.nn.relu(inp)
    tmp2 = -tf.nn.relu(-inp)
    tmp1 = tf.matmul(tmp1,weights1) 
    tmp2 = tf.matmul(tmp2,weights2)
    out = tmp1 + tmp2  + bias1
#    scale = tf.get_variable('biases',[1,1], initializer=tf.constant_initializer(1.0))
#    out = tf.nn.l2_normalize(out,1)*scale
    return out
    
def layer_orth3(inp, shape, isTrainVar, scope, bn = False, activation = True, drop = False, keep = 0.5):
    weights1, bias1, weights2 = get_orth_weights3(shape)
    if activation:
        tmp1 = tf.nn.relu(inp)
        tmp2 = inp
    tmp1 = tf.matmul(tmp1,weights1) 
    tmp2 = tf.matmul(tmp2,weights2)
    out = tmp1 + tmp2  + bias1
    return out

def layer_orth(inp, shape, isTrainVar, scope, bn = False, activation = False):
    weights1, bias1 = get_orth_weights(shape)
    out = tf.matmul(inp,weights1) + bias1
#    if bn:
#        out = batchnorm(out, scope = scope)
    if activation:
        out = activation(out)
    return out    
    
def layer_reg(inp, shape, isTrainVar, scope, bn = False, activation = False):
    weights1, bias1 = get_weights(shape)
    out = tf.matmul(inp,weights1) + bias1
#    if bn:
#        out = batchnorm(out, scope = scope)
    if activation:
        out = activation(out)
    return out    


def layer_reg2(inp, shape, isTrainVar, scope, bn = False, activation = False):
    weights1, weights2, bias1 = get_weights(shape,double = True)
    tmp1 = tf.nn.relu(inp)
    tmp2 = -tf.nn.relu(-inp)
    tmp1 = tf.matmul(tmp1,weights1) 
    tmp2 = tf.matmul(tmp2,weights2)
    out = tmp1 + tmp2  + bias1
    return out
    
    

                  



def model(inp, init_dim, dim, num_of_labels, depth, isTrainVar, bn = 0, norm_start = 0, norm_end = 0):
#    dim = 128
#    depth = 30
#    inp = tf.nn.l2_normalize(inp,axis = 1)
    with tf.variable_scope('net_vars') as scope:
        with tf.variable_scope('input_block0') as scope:
            tmp = layer_reg(inp, [init_dim,dim], isTrainVar, scope)
            if norm_start==1:
                tmp = tf.nn.l2_normalize(tmp,axis = 1)
#            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.95), lambda: tmp)
        for i in range(depth - 1):     
            with tf.variable_scope('input_block_'+str(i+1)) as scope:
                tmp = layer_reg2(tmp, [dim,dim], isTrainVar, scope)
        
#                tmp = batchnorm(tmp, is_training=isTrainVar, scope='final_bn')
#                tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.95), lambda: tmp)
        with tf.variable_scope('input_final') as scope:
            if bn==1:
                tmp = batchnorm(tmp, is_training=isTrainVar, scope='final_bn')
            if norm_end==1:
                tmp = tf.nn.l2_normalize(tmp,axis = 1)
#            tmp = tf.nn.relu(tmp)
#            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.8), lambda: tmp)
            logits = layer_reg(tmp, [dim,num_of_labels], isTrainVar, scope, bn = False, activation = False)
    return logits

def model2(inp, init_dim, dim, num_of_labels, depth, isTrainVar, bn = 0, norm_start = 0, norm_end = 0):
#    dim = 128
#    depth = 30
#    inp = tf.nn.l2_normalize(inp,axis = 1)
    with tf.variable_scope('net_vars') as scope:
        with tf.variable_scope('input_block0') as scope:
            tmp = layer_orth(inp, [init_dim,dim], isTrainVar, scope)
            if norm_start==1:
                tmp = tf.nn.l2_normalize(tmp,axis = 1)
#            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.95), lambda: tmp)
        for i in range(depth - 1):     
            with tf.variable_scope('input_block_'+str(i+1)) as scope:
                tmp = layer_orth2(tmp, [dim,dim], isTrainVar, scope)
        
#                tmp = batchnorm(tmp, is_training=isTrainVar, scope='final_bn')
#                tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.95), lambda: tmp)
        with tf.variable_scope('input_final') as scope:
            if bn==1:
               tmp = batchnorm(tmp, is_training=isTrainVar, scope='final_bn')
            if norm_end==1:
               tmp = tf.nn.l2_normalize(tmp,axis = 1)
#            tmp = tf.nn.relu(tmp)
#            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.8), lambda: tmp)
            logits = layer_orth(tmp, [dim,num_of_labels], isTrainVar, scope, bn = False, activation = False)
    return logits
    


    
def reg_architecture(inp, init_dim, dim, num_of_labels, depth, isTrainVar, activation, bn = 0):
    with tf.variable_scope('net_vars') as scope:
        with tf.variable_scope('input_block0') as scope:
            tmp = layer_orth(inp, [init_dim,dim], isTrainVar, scope, activation = activation)
        for i in range(depth - 1):     
            with tf.variable_scope('input_block_'+str(i+1)) as scope:
                tmp = layer_orth(tmp, [dim,dim], isTrainVar, scope, activation = activation)

        with tf.variable_scope('input_final') as scope:
            logits = layer_orth(tmp, [dim,dim], num_of_labels, scope)
    return logits


def linear(input_, output_size, scope=None, bn = False, activation = None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
#        mat = np.random.normal(size = shape)
#        U, S, V = np.linalg.svd(np.matmul(mat,mat.T))
#        U = U[0:shape[0],0:shape[1]]
#        U = U.astype('float32')
#        matrix = tf.get_variable('weight1s',initializer = U)
        matrix = tf.get_variable(scope + "Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))

        tf.add_to_collection('l2_loss',(tf.nn.l2_loss(matrix)))
        
        tf.add_to_collection('weights',(matrix))
        bias = tf.get_variable(scope + "bias", [output_size],
            initializer=tf.constant_initializer(0.0))
        tf.add_to_collection('weights',(bias))
        output = tf.matmul(input_, matrix) + bias                
        if bn:
            output = batchnorm(output, scope = scope)
        if activation:
            output = activation(output)
        return output     




    
def res_architecture(inp, dim, num_of_labels, activation, depth, isTrain_node, bn = 0):
    tmp = linear(inp, dim, scope='init', bn = bn)
#    tmp = tf.cond(isTrain_node, lambda: tf.contrib.nn.alpha_dropout(tmp, 0.95), lambda: tmp)
    for i in range(int(depth/2) - 1):
        tmp = res(tmp, dim, scope='layer_'+str(i), bn = bn, activation = activation)
#        tmp = tf.cond(isTrain_node, lambda: tf.contrib.nn.alpha_dropout(tmp, 0.95), lambda: tmp)
#    tmp = tf.cond(isTrain_node, lambda: tf.nn.dropout(tmp, 1.0), lambda: tmp)
#    if bn==1:
#        tmp = batchnorm(tmp, is_training=isTrain_node, scope='final_bn')
    out = linear(tmp, num_of_labels, scope='out', bn = 0)   
    return out
    
def res_architecture2(inp, dim, num_of_labels, activation, depth, isTrain_node, bn = 0):
    tmp = linear(inp, dim, scope='init', bn = bn)
#    tmp = tf.cond(isTrain_node, lambda: tf.contrib.nn.alpha_dropout(tmp, 0.95), lambda: tmp)
    for i in range(int(depth/2) - 1):
        tmp = res2(tmp, dim, scope='layer_'+str(i), bn = bn, activation = activation)
#        tmp = tf.cond(isTrain_node, lambda: tf.contrib.nn.alpha_dropout(tmp, 0.95), lambda: tmp)
#    tmp = tf.cond(isTrain_node, lambda: tf.nn.dropout(tmp, 1.0), lambda: tmp)
#    if bn==1:
#        tmp = batchnorm(tmp, is_training=isTrain_node, scope='final_bn')
    out = linear(tmp, num_of_labels, scope='out', bn = 0)   
    return out
    
    
def res(input_, output_size, scope=None, bn = False, activation = None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix1 = tf.get_variable(scope + "Matrix1", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.1/np.sqrt(shape[1])))
        matrix2 = tf.get_variable(scope + "Matrix2", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.1/np.sqrt(shape[1])))                        

        bias1 = tf.get_variable(scope + "bias1", [output_size],
            initializer=tf.constant_initializer(0.0))
        bias2 = tf.get_variable(scope + "bias2", [output_size],
            initializer=tf.constant_initializer(0.0))
        
        tmp = tf.matmul(input_, matrix1) + bias1  
#        if bn==1:
#            with tf.variable_scope('bn1'):
#                tmp = batchnorm(tmp, scope = scope)  
        tmp = tf.nn.relu(tmp)
        tmp = tf.matmul(tmp, matrix2) + bias2 
#        if bn==1:
#            with tf.variable_scope('bn2'):  
#                tmp = batchnorm(tmp, scope = scope)  
        output = tmp + input_
        return output     
        
def res2(input_, output_size, scope=None, bn = False, activation = None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix1 = tf.get_variable(scope + "Matrix1", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))
        matrix2 = tf.get_variable(scope + "Matrix2", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))                        

        bias1 = tf.get_variable(scope + "bias1", [output_size],
            initializer=tf.constant_initializer(0.0))
        bias2 = tf.get_variable(scope + "bias2", [output_size],
            initializer=tf.constant_initializer(0.0))
        
        tmp = tf.matmul(input_, matrix1) + bias1  
#        if bn==1:
        with tf.variable_scope('bn1'):
            tmp = batchnorm(tmp, scope = scope)  
        tmp = tf.nn.relu(tmp)
        tmp = tf.matmul(tmp, matrix2) + bias2 
#        if bn==1:
        with tf.variable_scope('bn2'):  
            tmp = batchnorm(tmp, scope = scope)  
        output = tmp + input_
        return output     


        
  