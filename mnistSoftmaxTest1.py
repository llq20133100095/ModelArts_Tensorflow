#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:17:57 2017

@author: llq

create a softmax
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

"""
Input parameters:
  python mnistSoftmaxTest1.py --data_url=./MNIST_data --train_url=./ --is_training=True
"""
tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Train Url')
tf.flags.DEFINE_integer('max_num_steps', 1000, 'training epochs')
tf.flags.DEFINE_boolean('is_training', True, 'train')
tf.flags.DEFINE_string('num_gpus', None, 'train')


flags = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = flags.num_gpus
checkpoint = os.path.join(flags.train_url, 'checkpoint/model.ckpt')
logs = os.path.join(flags.train_url, 'mnist_logs/')

def main(*args):
    #import the data
    mnist = input_data.read_data_sets(flags.data_url, one_hot=True)
    
    """
    1.create the x,w,b,y
    """
    # x:input images.
    #this tensor is [None,784]
    x=tf.placeholder(tf.float32,[None,784])
    
    #create the W and b
    W=tf.Variable(tf.zeros([784,10]))
    b=tf.Variable(tf.zeros([10]))
    
    #create softmax
    y=tf.nn.softmax(tf.matmul(x,W)+b) 
    
    
    """
    2.train the model,computate the cross_entropy
    """
    #the true label
    y_=tf.placeholder("float",[None,10])
    
    #cross_entropy
    cross_entropy=-tf.reduce_sum(y_*tf.log(y))
    
    #Gradient Descent;
    #the learning rate=0.01
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    
    """
    3.assess the model
    """
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy) 
    merged_summary_op = tf.summary.merge_all() 
    
    """
    4.init the variable, saver
    """
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    #train
    if flags.is_training:
        #start the Session
        with tf.Session() as sess:
            sess.run(init)
            
            #5.tensorboard

            summary_writer = tf.summary.FileWriter(logs, sess.graph)
    
    
            """
            start to train the model.
            train 1000 iteration
            """
            for i in range(flags.max_num_steps):
                batch_xs,batch_ys=mnist.train.next_batch(100)  
                sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
                
                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y_: batch_ys})
                summary_writer.add_summary(summary_str, i)
                
                #save the variable in each 100 iter
                if(i % 100 == 0):
                    saver.save(sess, checkpoint)
                    print("iter: %d" % (i))
                    print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
            
            summary_writer.close()
    else:
        #test 
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, checkpoint)
            print("finally predicte:")
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
if __name__== '__main__':

    tf.app.run(main=main)
