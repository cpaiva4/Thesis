import numpy as np
import tensorflow as tf2
tf=tf2.compat.v1
tf.disable_v2_behavior()

def convimp(x, W, b, strides=1,strides2=8,pad="VALID"):
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides2, 1], padding=pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    #  input_layer = tf.reshape(data, [batch_size , image_height , image_width,channels])

def maxpoolimp(x, k=2,k2=32,pad="VALID"):
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k2, 1],strides=[1, k, k, k2, 1], padding=pad)

def cost_weights(labels,batch_size,cost):
    weights=np.ones(batch_size)
    weights[labels[:,1]==0]=1
    weights[labels[:,1] == 1]=cost
    return weights

def cost_weights2(labels):
    weights=0.4
    if labels[0]==0:
        weights=350
    return weights


def retrieve_batch(channels,batch_size,lab):
    maps=np.zeros((batch_size,5,5,1280))
    l=0
    cut=-1
    for i in range(len(lab)-4):
        #print(len(lab)-4)
        if lab[i,1]!=lab[i+4,1]:
            l=4
            cut=i
            #l=(lab[i,1]!=lab[i+1,1])+(lab[i,1]!=lab[i+2,1])+(lab[i,1]!=lab[i+3,1])+(lab[i,1]!=lab[i+4,1])
        for j in range(25):
            maps[i,j//5,j%5,:]=channels[j,(i+l)*256:(i+l)*256+1280] #256=1 second step
    maps=np.reshape(maps,[batch_size,5,5,1280,1])
    #print(maps[14])
    return maps,l,cut

def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

def bias_init(shape):
    return tf.Variable(tf.zeros(shape=shape))

def dense_layer(input, in_size, out_size, dropout=False, activation=tf.nn.relu):
    weights = weights_init([in_size, out_size])
    bias = bias_init([out_size])

    layer = tf.matmul(input, weights) + bias

    if activation != None:
        layer = activation(layer)

    if dropout:
        layer = tf.nn.dropout(layer, 0.5)

    return layer

def flatten(layer, batch_size, seq_len):
    dims = layer.get_shape()
    number_of_elements = dims[2:].num_elements()

    reshaped_layer = tf.reshape(layer, [batch_size, int(seq_len / 2), number_of_elements])
    return reshaped_layer, number_of_elements

