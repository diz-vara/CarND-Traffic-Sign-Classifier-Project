#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:46:21 2017

@author: avarfolomeev
"""
#%%
tf.reset_default_graph();
keep_prob = tf.placeholder(tf.float32, name='keep_prob')                                           

batch_x = tf.placeholder(tf.float32, [None,32,32,3], name='batch_x')
batch_y = tf.placeholder(tf.int32, (None),name='batch_y')
ohy = tf.one_hot(batch_y,43,name='one_hot');
fc2 = MixNet(batch_x)

step = tf.Variable(0, trainable=False,name='step')
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, step, 
                                          70, 0.998, staircase=True)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(loss_op, global_step = step)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ohy, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

e_t = tf.where(tf.not_equal(tf.cast(tf.argmax(fc2, 1),tf.int32), batch_y))


saver = tf.train.Saver();

#%%


def test_data(xv, yv):
    cum_errors = np.empty(0,np.int32)
    steps_per_epoch = np.int(np.floor(xv.shape[0] // BATCH_SIZE))
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_start = step * BATCH_SIZE
        bx = xv[batch_start:batch_start + BATCH_SIZE]
        by = yv[batch_start:batch_start + BATCH_SIZE]
        loss, acc, err = sess.run([loss_op, accuracy_op, e_t], feed_dict={batch_x : bx, batch_y: by, keep_prob: 1.0})
        #print(err)
        cum_errors = np.append(cum_errors,(err+batch_start));
        total_acc += (acc * bx.shape[0])
        total_loss += (loss * bx.shape[0])
        
    return cum_errors, total_loss/num_examples, total_acc/num_examples



#%%

print ("testing " , save_file)

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)


    e, val_loss, val_acc = test_data(Xgn_test,y_test)
    print("Validation loss = {:.3f}".format(val_loss))
    print("Validation accuracy = {:.3f}".format(val_acc))

    # Evaluate on the test data


