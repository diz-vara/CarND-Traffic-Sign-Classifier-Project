#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:46:21 2017

@author: avarfolomeev
"""
#%%
#batch_x = tf.placeholder(tf.float32, [None,32,32,3])
#batch_y = tf.placeholder(tf.int32, [None])
#keep_prob = tf.placeholder(tf.float32)   
#ohy = tf.one_hot(batch_y,43);
#fc2 = MixNet(batch_x)

#%%
corr_pred = tf.equal(tf.cast(tf.argmax(fc2, 1),tf.int32), batch_y)
e_t = tf.where(tf.not_equal(tf.cast(tf.argmax(fc2, 1),tf.int32), batch_y))
acc_op = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
lss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))



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
        loss, acc, err = sess.run([lss_op, acc_op, e_t], feed_dict={batch_x : bx, batch_y: by, keep_prob: 1.0})
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


    e, val_loss, val_acc = test_data(xval,yval)
    print("Validation loss = {:.3f}".format(val_loss))
    print("Validation accuracy = {:.3f}".format(val_acc))

    # Evaluate on the test data


