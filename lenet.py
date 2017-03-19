"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
#%%
import tensorflow as tf
from tensorflow.contrib.layers import flatten


# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    
    #32x32x1 -> 28x28x6
    w1 = tf.Variable(tf.truncated_normal((5,5,3,6),0,0.01))
    b1 = tf.Variable(tf.truncated_normal([6],0,0.001))

    c1 = tf.nn.conv2d(x,w1, strides = [1,1,1,1], padding='VALID') + b1
    c1 = tf.nn.relu(c1)
    #28x28x6 -> 14x14x6
    c1 = tf.nn.max_pool(c1, (1,2,2,1), (1,2,2,1), padding='VALID')
    
    #14x14x6 -> 10x10x16
    w2 = tf.Variable(tf.truncated_normal((5,5,6,16),0,0.01))
    b2 = tf.Variable(tf.truncated_normal([16], 0,0.001))

    c2 = tf.nn.conv2d(c1,w2, strides = [1,1,1,1], padding='VALID') + b2
    c2 = tf.nn.relu(c2)
    # -> 5x5x16
    c2 = tf.nn.max_pool(c2, (1,2,2,1), (1,2,2,1), padding='VALID')

    #400
    f0 = flatten(c2)
    w3 = tf.Variable(tf.truncated_normal((400,120),0,0.01))
    b3 = tf.Variable(tf.truncated_normal([120],0,0.001))

    f1 = tf.matmul(f0,w3) + b3
    f1 = tf.nn.relu(f1)
    
    w4 = tf.Variable(tf.truncated_normal((120,86),0,0.01))
    b4 = tf.Variable(tf.truncated_normal([86],0,0.001))
    
    f2 = tf.matmul(f1,w4) + b4
    f2 = tf.nn.relu(f2)
    
    w5 = tf.Variable(tf.truncated_normal((86,43),0,0.01))
    b5 = tf.Variable(tf.truncated_normal([43],0,0.001))
    
    logits = tf.matmul(f2,w5) + b5


    return logits
