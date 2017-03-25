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
from tensorflow.contrib.layers import flatten

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
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={batch_x : bx, batch_y: by, keep_prob: 1.0})
        total_acc += (acc * bx.shape[0])
        total_loss += (loss * bx.shape[0])
    return total_loss/num_examples, total_acc/num_examples


#%%
EPOCHS = 50
BATCH_SIZE = 100


keep_prob = tf.placeholder(tf.float32)                                           


#sigs are 32x32x3
batch_x = tf.placeholder(tf.float32, [None,32,32,3])
# 32 types
batch_y = tf.placeholder(tf.int32, (None))
ohy = tf.one_hot(batch_y,43);
fc2 = MixNet(batch_x)

step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, step, 
                                          50, 0.998, staircase=True)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(loss_op, global_step = step)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ohy, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver();


#%%
    
FOLDS = 5;
EPOCHS=20;
#config = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
with tf.Session() as sess:

    # Train model


    loss = 0

    s_val_loss = [];
    s_val_acc = [];
    
    print ('training net', save_file);
    for fold in range(FOLDS):
        sess.run(tf.global_variables_initializer())
        idxVal,idxTrn = splitIndicies(devIndicies,20,fold)
        (x,y) = augmentImageList(Xgn[idxTrn], Y[idxTrn], 5000)
        xval = Xgn[idxVal]
        yval = Y[idxVal]
        idx = np.arange(x.shape[0])
        steps_per_epoch = np.int32(x.shape[0] // BATCH_SIZE)
        num_examples = steps_per_epoch * BATCH_SIZE
        
        
        for i in range(EPOCHS):
            print(save_file,"fold {}".format(fold+1), "EPOCH {} ...".format(i+1))
            np.random.shuffle(idx)
            for step in range(steps_per_epoch):
                batch_start = step * BATCH_SIZE
                bx = x[idx[batch_start:batch_start + BATCH_SIZE]]
                by = y[idx[batch_start:batch_start + BATCH_SIZE]]
    
                _,loss = sess.run([train_op, loss_op], feed_dict={batch_x: bx, batch_y: by, keep_prob: 0.5})
                #print ("Epoch ", "%4d" % i, " ,step ", "%4d" % step, " from ", "%4d" % steps_per_epoch, "\r");
    
            val_loss, val_acc = eval_data(xval, yval)
            print("Validation loss = {:.3f}".format(val_loss), "Train loss = {:.5f}".format(loss))
            print("Validation accuracy = {:.3f}".format(val_acc), " Learning rate", "%.9f" % sess.run(learning_rate))
            print()
    
        s_val_loss.append(val_loss);
        s_val_acc.append(val_acc);
        print("average loss = {:.5f}".format(sum(s_val_loss)/len(s_val_loss)), "average accuracy = {:.5f}".format(sum(s_val_acc)/len(s_val_acc)))

    print("average loss = {:.5f}".format(sum(s_val_loss)/len(s_val_loss)), "average accuracy = {:.5f}".format(sum(s_val_acc)/len(s_val_acc)))
    saver.save(sess,save_file)    

    # Evaluate on the test data


