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
from tensorflow.contrib.layers import batch_norm

save_file = './mixNet0_Named.ckpt'

# MixNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
def MixNet(x):
    s = 0.2
    
    #32x32x3 -> 30x30x4
    w11 = tf.Variable(tf.truncated_normal((3,3,3,4),0,s),'w11')
    b11 = tf.Variable(tf.truncated_normal([4],0,0.001),'b11')

    c1 = tf.nn.conv2d(x,w11, strides = [1,1,1,1], padding='VALID') + b11
    c1 = tf.nn.relu(c1)


    
    #30x30x4 -> 28x28x8
    w12 = tf.Variable(tf.truncated_normal((3,3,4,8),0,s),'w12')
    b12 = tf.Variable(tf.truncated_normal([8],0,0.001),'b12')
    
    c1 = tf.nn.conv2d(c1,w12, strides = [1,1,1,1], padding='VALID') + b12
    
    #28x28x8 -> 14x14x8
    c1 = tf.nn.max_pool(c1, (1,2,2,1), (1,2,2,1), padding='VALID')
    c1 = tf.nn.relu(c1)

    #ac1 = tf.nn.avg_pool(c1,(1,4,4,1),(1,4,4,1), padding='SAME')
    #flat1 = flatten(ac1);    
    #print("layer1 :",c1.get_shape(),ac1.get_shape(),"; flattened=", flat1.get_shape())
    #c1 = tf.nn.dropout(c1, keep_prob);
    
    #14x14x16 -> 12x12x8
    w21 = tf.Variable(tf.truncated_normal((3,3,8,8),0,s),'w21')
    b21 = tf.Variable(tf.truncated_normal([8],0,0.001),'b21')

    c2 = tf.nn.conv2d(c1,w21, strides = [1,1,1,1], padding='VALID') + b21
    c2 = tf.nn.relu(c2)
    
    
    #12x12x16 -> 10x10x16
    w22 = tf.Variable(tf.truncated_normal((3,3,8,16),0,s),'w22')
    b22 = tf.Variable(tf.truncated_normal([16],0,0.01),'b22')
    
    c2 = tf.nn.conv2d(c2,w22, strides = [1,1,1,1], padding='VALID') + b22
    
    #10x10x16 -> 5x5x16
    c2 = tf.nn.max_pool(c2, (1,2,2,1), (1,2,2,1), padding='VALID')
    c2 = tf.nn.relu(c2)

    #5X5X32->100
    #w23 = tf.Variable(tf.truncated_normal((1,1,16,4),0,0.1))
    #b23 = tf.Variable(tf.truncated_normal([4],0,0.01))
    #c23 = tf.nn.conv2d(c2,w23, strides = [1,1,1,1], padding='VALID') + b23
    flat2 = flatten(c2);    
    print("layer2 :",c2.get_shape(),c2.get_shape(),"; flattened=", flat2.get_shape())
    
    #c2 = tf.nn.dropout(c2, keep_prob)


    #5x5x16 -> 3x3x16
    w31 = tf.Variable(tf.truncated_normal((3,3,16,32),0,s),'w31')
    b31 = tf.Variable(tf.truncated_normal([32],0,0.01),'b31')

    c3 = tf.nn.conv2d(c2,w31, strides = [1,1,1,1], padding='VALID') + b31
    c3 = tf.nn.relu(c3)
    
    #3x3x32 -> 1x1x64
    w32 = tf.Variable(tf.truncated_normal((3,3,32,32),0,s),'w32')
    b32 = tf.Variable(tf.truncated_normal([32],0,0.01),'b32')

    c3 = tf.nn.conv2d(c3,w32, strides = [1,1,1,1], padding='VALID') + b32
    c3 = tf.nn.relu(c3)
    #1X256 -> 256
    flat3 = flatten(c3);    
    print("layer3 :",c3.get_shape(),"; flattened=", flat3.get_shape())
    
   
    #1568+800+128
    lin1 = tf.concat([flat2,flat3], 1)
    lin1len = int(lin1.get_shape()[1]);
    print("lin1 shape:",lin1.get_shape(), lin1len)

    #1568+800+128
    wl1 = tf.Variable(tf.truncated_normal((lin1len,120),0,s),'wl1')
    bl1 = tf.Variable(tf.truncated_normal([120],0,0.001),'bl1')

    lin1 = tf.nn.dropout(lin1, keep_prob)
    lin1 = tf.matmul(lin1,wl1) + bl1
    lin1 = tf.nn.elu(lin1)
    
    wl2 = tf.Variable(tf.truncated_normal((120,43),0,s),'wl2')
    bl2 = tf.Variable(tf.truncated_normal([43],0,0.001),'bl2')

    lin2 = tf.matmul(lin1,wl2) + bl2
    #lin2 = tf.nn.relu(lin2)
   
    return lin2;
    
