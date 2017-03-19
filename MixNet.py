"""
MixNet Architecture


    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
#%%
import tensorflow as tf
from tensorflow.contrib.layers import flatten


# MixNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
def MixNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    
    #32x32x3 -> 30x30x8
    w11 = tf.Variable(tf.truncated_normal((3,3,3,4),0,0.1))
    b11 = tf.Variable(tf.truncated_normal([4],0,0.001))

    c1 = tf.nn.conv2d(x,w11, strides = [1,1,1,1], padding='VALID') + b11
    c1 = tf.nn.relu(c1)
    
    #30x30x8 -> 28x28x8
    w12 = tf.Variable(tf.truncated_normal((3,3,4,16),0,0.1))
    b12 = tf.Variable(tf.truncated_normal([16],0,0.001))
    
    c1 = tf.nn.conv2d(c1,w12, strides = [1,1,1,1], padding='VALID') + b12
    
    #28x28x8 -> 14x14x8
    c1 = tf.nn.max_pool(c1, (1,2,2,1), (1,2,2,1), padding='VALID')
    c1 = tf.nn.relu(c1)
    
    #14x14x8 - > 1568
    flat1 = flatten(c1);    
    print("layer1 :",c1.get_shape(),"; flattened=", flat1.get_shape())
    
    #14x14x8 -> 12x12x16
    w21 = tf.Variable(tf.truncated_normal((3,3,16,16),0,0.1))
    b21 = tf.Variable(tf.truncated_normal([16],0,0.001))

    c2 = tf.nn.conv2d(c1,w21, strides = [1,1,1,1], padding='VALID') + b21
    c2 = tf.nn.relu(c2)
    
    #12x12x16 -> 10x10x32
    w22 = tf.Variable(tf.truncated_normal((3,3,16,64),0,0.1))
    b22 = tf.Variable(tf.truncated_normal([64],0,0.01))
    
    c2 = tf.nn.conv2d(c2,w22, strides = [1,1,1,1], padding='VALID') + b22
    
    #10x10x32 -> 5x5x32
    c2 = tf.nn.max_pool(c2, (1,2,2,1), (1,2,2,1), padding='VALID')
    c2 = tf.nn.relu(c2)
    #5X5X32->800
    flat2 = flatten(c2);    
    print("layer2 :",c2.get_shape(),"; flattened=", flat2.get_shape())
    


    #5x5x64 -> 3x3x64
    w31 = tf.Variable(tf.truncated_normal((3,3,64,128),0,0.1))
    b31 = tf.Variable(tf.truncated_normal([128],0,0.01))

    c3 = tf.nn.conv2d(c2,w31, strides = [1,1,1,1], padding='VALID') + b31
    c3 = tf.nn.relu(c3)
    
    #3x3x64 -> 1x1x512
    w32 = tf.Variable(tf.truncated_normal((3,3,128,512),0,0.1))
    b32 = tf.Variable(tf.truncated_normal([512],0,0.01))

    c3 = tf.nn.conv2d(c3,w32, strides = [1,1,1,1], padding='VALID') + b32
    c3 = tf.nn.relu(c3)
    #1X128 -> 512
    flat3 = flatten(c3);    
    print("layer3 :",c3.get_shape(),"; flattened=", flat3.get_shape())
    
   
    #1568+800+128
    lin1 = tf.concat([flat2,flat3], 1)
    lin1len = int(lin1.get_shape()[1]);
    print("lin1 shape:",lin1.get_shape(), lin1len)

    #1568+800+128
    wl1 = tf.Variable(tf.truncated_normal((lin1len,256),0,0.01))
    bl1 = tf.Variable(tf.truncated_normal([256],0,0.001))

    lin1 = tf.nn.dropout(lin1, keep_prob)
    lin1 = tf.matmul(lin1,wl1) + bl1
    lin1 = tf.nn.relu(lin1)
    
    wl2 = tf.Variable(tf.truncated_normal((256,43),0,0.01))
    bl2 = tf.Variable(tf.truncated_normal([43],0,0.001))

    lin2 = tf.matmul(lin1,wl2) + bl2
    #lin2 = tf.nn.relu(lin2)
   
    return lin2;
    
