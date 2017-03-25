#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:19:38 2017

@author: avarfolomeev
"""

#%%
EPOCHS=30

Xgn_train = normalizeImageList(X_train,'G')
Xgn_valid = normalizeImageList(X_valid,'G')


(x,y) = augmentImageList(Xgn_train, y_train, 10000)
xval = Xgn_valid
yval = y_valid

tf.reset_default_graph();

keep_prob = tf.placeholder(tf.float32, name='keep_prob')                                           

batch_x = tf.placeholder(tf.float32, [None,32,32,3], name='batch_x')
batch_y = tf.placeholder(tf.int32, (None),name='batch_y')
ohy = tf.one_hot(batch_y,43,name='one_hot');
fc2 = MixNet(batch_x)

step = tf.Variable(0, trainable=False,name='step')
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, step, 
                                          100, 0.998, staircase=True)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(loss_op, global_step = step)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ohy, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
saver = tf.train.Saver();

final_save_file = save_file;

with tf.Session() as sess:

    # Train model
    sess.run(tf.global_variables_initializer())

    loss = 0
    

    
    idx = np.arange(x.shape[0])
    steps_per_epoch = np.int32(x.shape[0] // BATCH_SIZE)
    num_examples = steps_per_epoch * BATCH_SIZE
        
        
    for i in range(EPOCHS):
        np.random.shuffle(idx)
        for step in range(steps_per_epoch):
            batch_start = step * BATCH_SIZE
            bx = x[idx[batch_start:batch_start + BATCH_SIZE]]
            by = y[idx[batch_start:batch_start + BATCH_SIZE]]
    
            _,loss = sess.run([train_op, loss_op], feed_dict={batch_x: bx, batch_y: by, keep_prob: 0.5})
    
        val_loss, val_acc = eval_data(xval, yval)
        trn_loss, trn_acc = eval_data(x, y)
        print(save_file, "EPOCH {} ...".format(i+1), 
              "Learning rate", "%.9f" % sess.run(learning_rate))
        print("Validation loss = {:.3f}".format(val_loss), 
              "Validation accuracy = {:.3f}".format(val_acc))
        print("Train loss (drop) = {:.5f}".format(loss), 
              "Train loss = {:.5f}".format(trn_loss), 
              "Train acc  = {:.5f}".format(trn_acc) )
        print()
    
    saver.save(sess,final_save_file)    
    
    # Evaluate on the test data
    tst_loss, tst_acc = eval_data(Xgn_test, y_test)
    print(save_file, "Test loss = {:.3f}".format(tst_loss), "Test accuracy = {:.3f}".format(tst_acc))



