#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:30:39 2019

@author: dcase
"""
import tensorflow as tf
from keras.layers import Input, Dense, TimeDistributed
from autopool import AutoPool1D
from keras import regularizers

class Conv_block(object):
    def __init__(self,kernel_size,layer_depth):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = tf.layers.conv2d(inputs=self.input_tensor,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bn1 = tf.layers.batch_normalization(conv1,training=self.is_training)
        relu1 = tf.nn.leaky_relu(bn1)
        
        conv2 = tf.layers.conv2d(inputs=relu1,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bn2 = tf.layers.batch_normalization(conv2,training=self.is_training)
        relu2 = tf.nn.leaky_relu(bn2)
       
        return relu2

class Res_block(object):
    def __init__(self,kernel_size,layer_depth):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = tf.layers.conv2d(inputs=self.input_tensor,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bn1 = tf.layers.batch_normalization(conv1,training=self.is_training)
        relu1 = tf.nn.leaky_relu(bn1)
        
        conv2 = tf.layers.conv2d(inputs=relu1,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bn2 = tf.layers.batch_normalization(conv2,training=self.is_training)
        
        conv1_1 = tf.layers.conv2d(inputs=self.input_tensor,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bn1_1 = tf.layers.batch_normalization(conv1_1,training=self.is_training)
        relu1_1 = tf.nn.leaky_relu(bn1_1)
        
        conv2_1 = tf.layers.conv2d(inputs=relu1_1,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bn2_1 = tf.layers.batch_normalization(conv2_1,training=self.is_training)
    
        relu2 = tf.nn.leaky_relu(bn2+bn2_1)
       
        return relu2

class CNN7_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num,hidden_layer_size):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        self.hidden_layer_size = hidden_layer_size
        self.conv_block1 = Conv_block(kernel_size,layer_depth[0])
        self.conv_block2 = Conv_block(kernel_size,layer_depth[1])
        self.conv_block3 = Conv_block(kernel_size,layer_depth[2])
 
    def forward(self,input_tensor,input_meta,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        self.input_meta = input_meta
        conv1 = self.conv_block1.forward(self.input_tensor,self.is_training)
        pool1 = tf.layers.average_pooling2d(conv1,pool_size=2,strides=2,padding='VALID')        
        conv2 = self.conv_block2.forward(pool1,self.is_training)
        pool2 = tf.layers.average_pooling2d(conv2,pool_size=2,strides=2,padding='VALID')        
        conv3 = self.conv_block3.forward(pool2,self.is_training)
        pool3 = tf.layers.average_pooling2d(conv3,pool_size=2,strides=2,padding='VALID')

        pool4 = tf.reduce_mean(pool3,axis=2)
###########         
        fea_frames = tf.shape(pool4)[1]        
        self.input_meta = tf.expand_dims(self.input_meta,1)
        self.input_meta = tf.tile(self.input_meta,[1,fea_frames,1])
        pool4 = tf.concat([pool4,self.input_meta],axis=-1)
        repr_size = tf.shape(pool4)[2]

        pool4 = TimeDistributed(Dense(self.hidden_layer_size, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            input_shape=(fea_frames, repr_size))(pool4)
        repr_size = self.hidden_layer_size


        # Output layer
        pool4 = TimeDistributed(Dense(self.classes_num,
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            name='output_t',
                            input_shape=(fea_frames, repr_size))(pool4)

            
        # Apply autopool over time dimension
        # y = AutoPool1D(kernel_constraint=keras.constraints.non_neg(),
        #                axis=1, name='output')(y)
        output = AutoPool1D(axis=1, name='output')(pool4)    ###(batch,num_classes)
        
        return output    
    
class CNN9_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num,hidden_layer_size):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        self.hidden_layer_size = hidden_layer_size
        self.conv_block1 = Conv_block(kernel_size,layer_depth[0])
        self.conv_block2 = Conv_block(kernel_size,layer_depth[1])
        self.conv_block3 = Conv_block(kernel_size,layer_depth[2])
        self.conv_block4 = Conv_block(kernel_size,layer_depth[3])
        # self.conv_block5 = Conv_block(kernel_size,layer_depth[3])
     
#    def transform_norm():
#        mean
        
    def forward(self,input_tensor,input_meta,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        self.input_meta = input_meta
        conv1 = self.conv_block1.forward(self.input_tensor,self.is_training)
        pool1 = tf.layers.average_pooling2d(conv1,pool_size=2,strides=2,padding='VALID')        
        conv2 = self.conv_block2.forward(pool1,self.is_training)
        pool2 = tf.layers.average_pooling2d(conv2,pool_size=2,strides=2,padding='VALID')        
        conv3 = self.conv_block3.forward(pool2,self.is_training)
        pool3 = tf.layers.average_pooling2d(conv3,pool_size=2,strides=2,padding='VALID')
        conv4 = self.conv_block4.forward(pool3,self.is_training)
        pool4 = tf.layers.average_pooling2d(conv4,pool_size=1,strides=1,padding='VALID')
#        conv5 = self.conv_block4.forward(pool4,self.is_training)
#        pool5 = tf.layers.average_pooling2d(conv5,pool_size=2,strides=2,padding='VALID')

        pool4 = tf.reduce_mean(pool4,axis=2)
###########         
        fea_frames = tf.shape(pool4)[1]        
        self.input_meta = tf.expand_dims(self.input_meta,1)
        self.input_meta = tf.tile(self.input_meta,[1,fea_frames,1])
        pool4 = tf.concat([pool4,self.input_meta],axis=-1)
        repr_size = tf.shape(pool4)[2]

        pool4 = TimeDistributed(Dense(self.hidden_layer_size, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            input_shape=(fea_frames, repr_size))(pool4)
        repr_size = self.hidden_layer_size


        # Output layer
        pool4 = TimeDistributed(Dense(self.classes_num,
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            name='output_t',
                            input_shape=(fea_frames, repr_size))(pool4)

            
        # Apply autopool over time dimension
        # y = AutoPool1D(kernel_constraint=keras.constraints.non_neg(),
        #                axis=1, name='output')(y)
        output = AutoPool1D(axis=1, name='output')(pool4)    ###(batch,num_classes)
###########

      
#        pool4 = tf.reduce_max(pool4,axis=1)
#        flatten = tf.layers.flatten(pool4)
#
#        flatten = tf.layers.flatten(pool4)
#        flatten = tf.concat([flatten,self.input_meta],axis=-1)
#        dense1 = tf.layers.dense(flatten,units=self.hidden_layer_size,
#                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),activation=tf.nn.leaky_relu)
#        output = tf.layers.dense(dense1,units=self.classes_num)
        
        return output    
    
class CRNN9_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num,hidden_layer_size):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        self.hidden_layer_size = hidden_layer_size
        self.conv_block1 = Conv_block(kernel_size,layer_depth[0])
        self.conv_block2 = Conv_block(kernel_size,layer_depth[1])
        self.conv_block3 = Conv_block(kernel_size,layer_depth[2])
        self.conv_block4 = Conv_block(kernel_size,layer_depth[3])
        # self.conv_block5 = Conv_block(kernel_size,layer_depth[3])
        
    def forward(self,input_tensor,input_meta,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        self.input_meta = input_meta
        conv1 = self.conv_block1.forward(self.input_tensor,self.is_training)
        pool1 = tf.layers.average_pooling2d(conv1,pool_size=2,strides=2,padding='VALID')        
        conv2 = self.conv_block2.forward(pool1,self.is_training)
        pool2 = tf.layers.average_pooling2d(conv2,pool_size=2,strides=2,padding='VALID')        
        conv3 = self.conv_block3.forward(pool2,self.is_training)
        pool3 = tf.layers.average_pooling2d(conv3,pool_size=2,strides=2,padding='VALID')
        conv4 = self.conv_block4.forward(pool3,self.is_training)
        pool4 = tf.layers.average_pooling2d(conv4,pool_size=2,strides=2,padding='VALID')
   

#####################
        
        fea_frames = pool4.get_shape().as_list()[1]
        fea_bins = pool4.get_shape().as_list()[2]
        reshaped = tf.reshape(pool4,[-1,fea_frames,fea_bins*self.layer_depth[3]])
        num_units  = [128]
        basic_cells = [tf.nn.rnn_cell.GRUCell(num_units=n) for n in num_units]
        cells = tf.nn.rnn_cell.MultiRNNCell(basic_cells,  state_is_tuple=True)
        (outputs,state) = tf.nn.dynamic_rnn(cells, reshaped, sequence_length=None,dtype=tf.float32,time_major=False)
        pool4 = tf.reshape(outputs,[-1,fea_frames,fea_bins,32])
  
###########   
        pool4 = tf.reduce_mean(pool4,axis=2)
###########
        
        fea_frames = tf.shape(pool4)[1]        
        self.input_meta = tf.expand_dims(self.input_meta,1)
        self.input_meta = tf.tile(self.input_meta,[1,fea_frames,1])
        pool4 = tf.concat([pool4,self.input_meta],axis=-1)
        repr_size = tf.shape(pool4)[2]

        pool4 = TimeDistributed(Dense(self.hidden_layer_size, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            input_shape=(fea_frames, repr_size))(pool4)
        repr_size = self.hidden_layer_size


        # Output layer
        pool4 = TimeDistributed(Dense(self.classes_num,
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            name='output_t',
                            input_shape=(fea_frames, repr_size))(pool4)

            
        # Apply autopool over time dimension
        # y = AutoPool1D(kernel_constraint=keras.constraints.non_neg(),
        #                axis=1, name='output')(y)
        output = AutoPool1D(axis=1, name='output')(pool4)    ###(batch,num_classes)
###########
      
#        reshaped = tf.reduce_mean(reshaped,axis=2)
#        reshaped = tf.reduce_max(reshaped,axis=1)
#        flatten = tf.layers.flatten(reshaped)
#        output = tf.layers.dense(flatten,units=self.classes_num)
        
        return output 

class CNN9_Res_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num,hidden_layer_size):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        self.hidden_layer_size = hidden_layer_size
        self.conv_block1 = Conv_block(kernel_size,layer_depth[0])
        self.conv_block2 = Conv_block(kernel_size,layer_depth[1])
        self.res_block3 = Res_block(kernel_size,layer_depth[2])
        self.res_block4 = Res_block(kernel_size,layer_depth[3])
        # self.conv_block5 = Conv_block(kernel_size,layer_depth[3])
     
#    def transform_norm():
#        mean
        
    def forward(self,input_tensor,input_meta,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        self.input_meta = input_meta
        conv1 = self.conv_block1.forward(self.input_tensor,self.is_training)
        pool1 = tf.layers.average_pooling2d(conv1,pool_size=2,strides=2,padding='VALID')        
        conv2 = self.conv_block2.forward(pool1,self.is_training)
        pool2 = tf.layers.average_pooling2d(conv2,pool_size=2,strides=2,padding='VALID')        
        conv3 = self.res_block3.forward(pool2,self.is_training)
        pool3 = tf.layers.average_pooling2d(conv3,pool_size=2,strides=2,padding='VALID')
        conv4 = self.res_block4.forward(pool3,self.is_training)
        pool4 = tf.layers.average_pooling2d(conv4,pool_size=1,strides=1,padding='VALID')
#        conv5 = self.conv_block4.forward(pool4,self.is_training)
#        pool5 = tf.layers.average_pooling2d(conv5,pool_size=2,strides=2,padding='VALID')

        pool4 = tf.reduce_mean(pool4,axis=2)
###########         
        fea_frames = tf.shape(pool4)[1]        
        self.input_meta = tf.expand_dims(self.input_meta,1)
        self.input_meta = tf.tile(self.input_meta,[1,fea_frames,1])
        pool4 = tf.concat([pool4,self.input_meta],axis=-1)
        repr_size = tf.shape(pool4)[2]

        pool4 = TimeDistributed(Dense(self.hidden_layer_size, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            input_shape=(fea_frames, repr_size))(pool4)
        repr_size = self.hidden_layer_size


        # Output layer
        pool4 = TimeDistributed(Dense(self.classes_num,
                                  kernel_regularizer=regularizers.l2(0.0001)),
                            name='output_t',
                            input_shape=(fea_frames, repr_size))(pool4)

            
        # Apply autopool over time dimension
        # y = AutoPool1D(kernel_constraint=keras.constraints.non_neg(),
        #                axis=1, name='output')(y)
        output = AutoPool1D(axis=1, name='output')(pool4)    ###(batch,num_classes)

        
        return output    
