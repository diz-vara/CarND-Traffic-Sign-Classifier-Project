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
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten


EPOCHS = 50
BATCH_SIZE = 50


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
#%%

x = targetX; #np.float32(X_train);
y = targetY;
xval = targetXv; #np.float32(X_valid);
yval = y_valid;


#sigs are 32x32x3
batch_x = tf.placeholder(tf.float32, [None,32,32,3])
# 32 types
batch_y = tf.placeholder(tf.int32, (None))
ohy = tf.one_hot(batch_y,43);
fc2 = LeNet(batch_x)

step = tf.Variable(0, trainable=False)
starter_learning_rate = 2e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, step, 
                                           500, 0.998, staircase=True)
                                           

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(loss_op, global_step = step)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ohy, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#%%

def eval_data(xv, yv):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = np.int(np.floor(xv.shape[0] // BATCH_SIZE))
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_start = step * BATCH_SIZE
        bx = xv[batch_start:batch_start + BATCH_SIZE]
        by = yv[batch_start:batch_start + BATCH_SIZE]
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={batch_x : bx, batch_y: by})
        total_acc += (acc * bx.shape[0])
        total_loss += (loss * bx.shape[0])
    return total_loss/num_examples, total_acc/num_examples




#%%
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps_per_epoch = np.int32(x.shape[0] // BATCH_SIZE)
    num_examples = steps_per_epoch * BATCH_SIZE

    idx = np.arange(x.shape[0])
    # Train model
    for i in range(EPOCHS):
        np.random.shuffle(idx)
        for step in range(steps_per_epoch):
            batch_start = step * BATCH_SIZE
            bx = x[idx[batch_start:batch_start + BATCH_SIZE]]
            by = y[idx[batch_start:batch_start + BATCH_SIZE]]

            loss = sess.run(train_op, feed_dict={batch_x: bx, batch_y: by})
            #print ("Epoch ", "%4d" % i, " ,step ", "%4d" % step, " from ", "%4d" % steps_per_epoch, "\r");

        val_loss, val_acc = eval_data(xval, yval)
        print("EPOCH {} ...".format(i+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.3f}".format(val_acc))
        print("Learning rate", "%.9f" % sess.run(learning_rate))
        print()

    # Evaluate on the test data


