#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:19:38 2017

@author: avarfolomeev
"""

EPOCHS=50


(x,y) = augmentImageList(Xgn, Y, 5000)
xval = Xgn_test
yval = y_test

saver = tf.train.Saver();

final_save_file = save_file + '.final1'

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
        print("EPOCH {} ...".format(i+1), 
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
    print("Test loss = {:.3f}".format(tst_loss), "Test accuracy = {:.3f}".format(tst_acc))



