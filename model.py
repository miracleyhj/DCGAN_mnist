#-*- coding: utf-8 -*-
import tensorflow as tf
from ops import *
BATCH_SIZE = 64

def generator(z,y,train=True):
    with tf.variable_scope('generator') as scope:
        yb = tf.reshape(y,[BATCH_SIZE,1,1,10],name='yb')
        # concat函数在不同版本里使用方式不同，在新版本里1与[z,y]的位置要互换
        #z=tf.concat(1,[z,y],name='z_concat_y')
        z = tf.concat([z,y],1,name='z_concat_y')
        h1 = tf.nn.relu(batch_norm_layer(fully_connected(z,1024,'g_fully_connected1'),
                                         is_train=train,name='g_bn1'))
        #h1 = tf.concat(1,[h1,y],name='active1_concat_y')
        h1 = tf.concat([h1,y],1,name='active1_concat_y')
        h2 = tf.nn.relu(batch_norm_layer(fully_connected(h1,128*49,'g_fully_connected2'),
                                         is_train=train,name='g_bn2'))
        h2 = tf.reshape(h2,[64,7,7,128],name='h2_reshape')
        h2 = conv_cond_concat(h2,yb,name='active2_concat_y')
        h3 = tf.nn.relu(batch_norm_layer(deconv2d(h2,[64,14,14,128],
                                                  name='g_deconv2d3'),
                                         is_train=train,name='g_bn3'))
        h3 = conv_cond_concat(h3,yb,name='active3_concat_y')
        h4 = tf.nn.sigmoid(deconv2d(h3,[64,28,28,1],
                                    name='g_deconv2d4'),name='generate_image')
        scope.reuse_variables()
    return h4
def discriminator(image,y):
    with tf.variable_scope('discriminator') as scope:
        yb = tf.reshape(y,[BATCH_SIZE,1,1,10],name='yb')
        x = conv_cond_concat(image,yb,name='image_concat_y')
        h1 = lrelu(conv2d(x,11,name='d_conv2d1'),name='lrelu1')
        h1 = conv_cond_concat(h1,yb,name='h1_concat_yb')
        h2 = lrelu(batch_norm_layer(conv2d(h1,74,name='d_conv2d2'),
                                    name='d_bn2'),name='lrelu2')
        h2 = tf.reshape(h2,[BATCH_SIZE,-1],name='reshape_lrelu2_to_2d')
        # h2 = tf.concat(1,[h2,y],name='lrelu2_concat_y')
        h2 = tf.concat([h2,y],1,name='lrelu2_concat_y')
        h3 = lrelu(batch_norm_layer(fully_connected(h2,1024,name='d_fully_connected3'),
                                    name='d_bn3'),name='lrelu3')
        # h3 = tf.concat(1,[h3,y],name='lrelu3_concat_y')
        h3 = tf.concat([h3,y],1,name='lrelu3_concat_y')
        h4 = fully_connected(h3,1,name='d_result_withouts_sigmoid')
        scope.reuse_variables()
    return tf.nn.sigmoid(h4,name='discriminator_result_with_sigmoid'),h4
def sampler(z,y,train=True):
    tf.get_variable_scope().reuse_variables()
    return generator(z,y,train=train)