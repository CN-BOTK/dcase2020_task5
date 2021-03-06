#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:57:27 2020

@author: dcase
"""

import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml
import random
import pickle as pk
import metrics
import tensorflow as tf
#import keras
#from keras.layers import Input, Dense, TimeDistributed
#from keras.models import Model
#from keras import regularizers
#from keras.optimizers import Adam
import keras.backend as K
from sklearn.preprocessing import StandardScaler
#from autopool import AutoPool1D
from models import CNN9_train,CRNN9_train,CNN7_train,CNN9_Res_train
from functions import calculate_scalar_of_tensor,scale

NUM_HOURS = 24
NUM_DAYS = 7
NUM_WEEKS = 52


## HELPERS

def load_train_data(file_list, train_file_idxs, feature_dir):
    
    train_data = []
    for train_id in train_file_idxs:
        data_path = os.path.join(feature_dir, file_list[train_id][:-4]+'.npy')
        train_data.append(np.load(data_path))
    train_data = np.asarray(train_data)
    
    return train_data

def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets

    Parameters
    ----------
    annotation_data

    Returns
    -------
    train_idxs
    valid_idxs

    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename', 'annotator_id']]\
                          .groupby(by=['split', 'audio_filename'], as_index=False)\
                          .min()\
                          .sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []

    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        # elif row['split'] == 'validate':
        elif row['split'] == 'validate' and row['annotator_id'] <= 0:
            # For validation examples, only use verified annotations
            valid_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)

def gen_train_batch(train_data,train_meta,train_label,batch_size):

    num = train_data.shape[0]
    arr = np.arange(num)
    np.random.shuffle(arr)
    train_data = train_data[arr]
    train_label = train_label[arr]
    for index in range(0,len(train_data),batch_size):       
        if (index+batch_size)<=(len(train_data)):
            excerpt = slice(index, index + batch_size)
        else:
            excerpt = slice(index,len(train_data))
        yield train_data[excerpt], train_meta[excerpt], train_label[excerpt]
    
def gen_val_batch(val_data, val_meta, batch_size):

    for index in range(0,len(val_data),batch_size):       
        if (index+batch_size)<=(len(val_data)):
            excerpt = slice(index, index + batch_size)
        else:
            excerpt = slice(index,len(val_data))
        yield val_data[excerpt], val_meta[excerpt]


def get_file_targets(annotation_data, labels):
    """
    Get file target annotation vector for the given set of labels

    Parameters
    ----------
    annotation_data
    labels

    Returns
    -------
    targets

    """
    file_list = annotation_data['audio_filename'].unique().tolist()
    count_dict = {fname: {label: 0 for label in labels} for fname in file_list}

    for _, row in annotation_data.iterrows():
        fname = row['audio_filename']
        split = row['split']
        ann_id = row['annotator_id']

        # For training set, only use crowdsourced annotations
        if split == "train" and ann_id <= 0:
            continue

        # For validate and test sets, only use the verified annotation
        # if split != "train" :
        if split != "train" and ann_id != 0:
            continue

        for label in labels:
            count_dict[fname][label] += row[label + '_presence']

    targets = np.array([[1.0 if count_dict[fname][label] > 0 else 0.0 for label in labels]
                        for fname in file_list])

    return targets

def generate_output_file(y_pred, file_idxs, results_dir, file_list, label_mode, taxonomy):
    """
    Write the output file containing model predictions

    Parameters
    ----------
    y_pred
    file_idxs
    results_dir
    file_list
    label_mode
    taxonomy

    Returns
    -------

    """
    output_path = os.path.join(results_dir, "output.csv")
    file_list = [file_list[idx] for idx in file_idxs]

    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                             for fine_id, fine_label in fine_dict.items()]
                           for coarse_id, fine_dict in taxonomy['fine'].items()]

    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                                          for fine_label in fine_list]
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(file_list, y_pred):
            row = [filename]

            if label_mode == "fine":
                fine_values = []
                coarse_values = [0 for _ in range(len(coarse_target_labels))]
                coarse_idx = 0
                fine_idx = 0
                for coarse_label, fine_label_list in zip(coarse_target_labels,
                                                         coarse_fine_labels):
                    for fine_label in fine_label_list:
                        if 'X' in fine_label.split('_')[0].split('-')[1]:
                            # Put a 0 for other, since the baseline doesn't
                            # account for it
                            fine_values.append(0.0)
                            continue

                        # Append the next fine prediction
                        fine_values.append(y[fine_idx])

                        # Add coarse level labels corresponding to fine level
                        # predictions. Obtain by taking the maximum from the
                        # fine level labels
                        coarse_values[coarse_idx] = max(coarse_values[coarse_idx],
                                                        y[fine_idx])
                        fine_idx += 1
                    coarse_idx += 1

                row += fine_values + coarse_values

            else:
                # Add placeholder values for fine level
                row += [0.0 for _ in range(len(full_fine_target_labels))]
                # Add coarse level labels
                row += list(y)

            csvwriter.writerow(row)


## DATA PREPARATION

def one_hot(idx, num_items):
    return [(0.0 if n != idx else 1.0) for n in range(num_items)]


def prepare_data(train_file_idxs, valid_file_idxs,
                 latitude_list, longitude_list, week_list, day_list, hour_list,
                 target_list, standardize=True):
    """
    Prepare inputs and targets for MIL training using training and validation indices.
    Parameters
    ----------
    train_file_idxs
    valid_file_idxs
    latitude_list
    longitude_list
    week_list
    day_list
    hour_list
    embeddings
    target_list
    standardize
    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler
    """


    X_train_loc = np.array([[[latitude_list[idx],
                             longitude_list[idx]]]
                            for idx in train_file_idxs])
    X_valid_loc = np.array([[[latitude_list[idx],
                             longitude_list[idx]]]
                            for idx in valid_file_idxs])

    X_train_time = np.array([
        [one_hot(week_list[idx] - 1, NUM_WEEKS) \
          + one_hot(day_list[idx], NUM_DAYS) \
          + one_hot(hour_list[idx], NUM_HOURS)]
        for idx in train_file_idxs])
    X_valid_time = np.array([
        [one_hot(week_list[idx] - 1, NUM_WEEKS) \
         + one_hot(day_list[idx], NUM_DAYS) \
         + one_hot(hour_list[idx], NUM_HOURS)]
        for idx in valid_file_idxs])

    X_train_cts = X_train_loc
    X_valid_cts = X_valid_loc

    y_train = np.array([target_list[idx] for idx in train_file_idxs])
    y_valid = np.array([target_list[idx] for idx in valid_file_idxs])

    if standardize:
        # Only standardize continuous valued inputs (embeddings + location)
        scaler = StandardScaler()
        scaler.fit(np.array([feat for feat_grp in X_train_cts for feat in feat_grp]))

        X_train_cts = np.array([scaler.transform(emb_grp) for emb_grp in X_train_cts])
        X_valid_cts = np.array([scaler.transform(emb_grp) for emb_grp in X_valid_cts])
    else:
        scaler = None

    # Concatenate all of the inputs
    X_train = np.concatenate((X_train_cts, X_train_time), axis=-1)
    X_valid = np.concatenate((X_valid_cts, X_valid_time), axis=-1)


    return X_train, y_train, X_valid, y_valid, scaler


## MODEL TRAINING


def train(annotation_path, taxonomy_path,  train_feature_dir, val_feature_dir,
          output_dir, load_checkpoint, load_checkpoint_path, 
          exp_id, label_mode, 
          batch_size=32, n_epochs=100, kernel_size=3, 
          layer_depth = [64,128,256,512], chs = 1, max_ckpt = 20,
          lr=1e-3, hidden_layer_size=256, snapshot = 5,
          num_hidden_layers=1, standardize=True,
          timestamp=None):
    """
    Train and evaluate a MIL MLP model.
    Parameters
    ----------
    annotation_path
    emb_dir
    output_dir
    label_mode
    batch_size
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
    standardize
    timestamp
    random_state

    Returns
    -------
    """


    # Load annotations and taxonomy
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    annotation_data_trunc = annotation_data[['audio_filename',
                                             'latitude',
                                             'longitude',
                                             'week',
                                             'day',
                                             'hour']].drop_duplicates()
    file_list = annotation_data_trunc['audio_filename'].to_list()
    latitude_list = annotation_data_trunc['latitude'].to_list()
    longitude_list = annotation_data_trunc['longitude'].to_list()
    week_list = annotation_data_trunc['week'].to_list()
    day_list = annotation_data_trunc['day'].to_list()
    hour_list = annotation_data_trunc['hour'].to_list()

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                          if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    print("* Preparing training data.")

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, valid_file_idxs = get_subset_split(annotation_data)

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
        num_classes = len(labels)
        y_true_num = len(full_fine_target_labels)
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
        num_classes = len(labels)
        y_true_num = num_classes
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    


    X_train_meta, y_train, X_valid_meta, y_valid_meta, scaler \
        = prepare_data(train_file_idxs, valid_file_idxs,
                       latitude_list, longitude_list,
                       week_list, day_list, hour_list,
                       target_list, standardize=standardize)
    
    print('X_train meta shape', X_train_meta.shape)
    print('y_train shape', y_train.shape)
    print('X_valid_meta shape', X_valid_meta.shape)
    print('y_valid shape', y_valid_meta.shape)
    
    meta_dims = X_train_meta.shape[2]
    
    
    X_train = load_train_data(file_list, train_file_idxs, train_feature_dir)
    X_valid = load_train_data(file_list, valid_file_idxs, val_feature_dir)
    _, frames, bins = X_train.shape
    print('X_train shape', X_train.shape)
    print('X_valid shape', X_valid.shape)
    
    (mean_train, std_train) = calculate_scalar_of_tensor(np.concatenate(X_train,axis=0))
    
    
    model = CNN9_Res_train(kernel_size,layer_depth,num_classes,hidden_layer_size)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    model_path = os.path.join(output_dir, 'exp'+exp_id)

    if scaler is not None:
        scaler_path = os.path.join(model_path, 'stdizer.pkl')
        with open(scaler_path, 'wb') as f:
            pk.dump(scaler, f)

    if label_mode == "fine":
        full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
        incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                                   for fine_dict in taxonomy['fine'].values()]
        coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                             for fine_dict in taxonomy['fine'].values()])

        # Create loss function that only adds loss for fine labels for which
        # the we don't have any incomplete labels
        def masked_loss(y_true, y_pred):
            loss = None
            for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
                true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
                true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]
                pred_end_idx = coarse_to_fine_end_idxs[coarse_idx]

                if coarse_idx != 0:
                    true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
                    pred_start_idx = coarse_to_fine_end_idxs[coarse_idx-1]
                else:
                    true_start_idx = 0
                    pred_start_idx = 0

                if true_incomplete_subidx is None:
                    true_end_idx = true_terminal_idx

                    sub_true = y_true[:, true_start_idx:true_end_idx]
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

                else:
                    # Don't include incomplete label
                    true_end_idx = true_terminal_idx - 1
                    true_incomplete_idx = true_incomplete_subidx + true_start_idx
                    assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
                    assert true_incomplete_idx == true_end_idx

                    # 1 if not incomplete, 0 if incomplete
                    mask = K.expand_dims(1 - y_true[:, true_incomplete_idx])

                    # Mask the target and predictions. If the mask is 0,
                    # all entries will be 0 and the BCE will be 0.
                    # This has the effect of masking the BCE for each fine
                    # label within a coarse label if an incomplete label exists
                    sub_true = y_true[:, true_start_idx:true_end_idx] * mask
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask

                if loss is not None:
                    loss += K.sum(K.binary_crossentropy(sub_true, sub_pred))
                else:
                    loss = K.sum(K.binary_crossentropy(sub_true, sub_pred))

            return loss
               
        loss_func = masked_loss
    else:
        
        def unmasked_loss(y_true, y_pred):
            
            loss = None
            loss = K.sum(K.binary_crossentropy(y_true, y_pred))
            return loss
        
        loss_func = unmasked_loss

    ###     placeholder
    x = tf.placeholder(tf.float32,shape=[None,frames,bins,chs],name='x')
    meta_x = tf.placeholder(tf.float32,shape=[None,meta_dims],name='meta_x')
    y = tf.placeholder(tf.float32,shape=[None,y_true_num],name='y')
    is_training = tf.placeholder(tf.bool,shape=None,name='is_training')
    
    ###     net output
    output = model.forward(input_tensor=x,input_meta=meta_x,is_training=is_training)
    sigmoid_output = tf.nn.sigmoid(output,name='sigmoid_output')
    loss = loss_func(y,sigmoid_output)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)            
    learning_rate = tf.Variable(float(lr), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
    with tf.control_dependencies(update_ops):        
#        train_op = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum).minimize(loss)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) 
            
    
    ###     start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=max_ckpt)
    sess=tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if load_checkpoint:
        saver.restore(sess,load_checkpoint_path)
    
    
    ###     tensorboard summary

    train_summary_dir = os.path.join(model_path, 'summaries', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)   
    
    loss_all=tf.placeholder(tf.float32,shape=None,name='loss_all')

    tf.add_to_collection("loss", loss_all)

    loss_summary = tf.summary.scalar('loss', loss_all)

    
    val_summary_dir = os.path.join(model_path, 'summaries', 'val')
    val_micro_auprc_summary_writer = tf.summary.FileWriter(os.path.join(val_summary_dir,'micro_auprc'), sess.graph)
    val_macro_auprc_summary_writer = tf.summary.FileWriter(os.path.join(val_summary_dir,'macro_auprc'), sess.graph)
    val_val_micro_F1score_summary_writer = tf.summary.FileWriter(os.path.join(val_summary_dir,'micro_F1score'), sess.graph)
    val_summary = tf.placeholder(tf.float32,shape=None,name='loss_all')
    tf.add_to_collection("val_summary", val_summary)
    val_summary_op = tf.summary.scalar('val_summary', val_summary)

    ###     train loop
    print("* Training model.")
    class_auprc_dict = {}
    for epoch in range(n_epochs):
        train_loss = 0 ; n_batch = 0 
        for X_train_batch, X_meta_batch, y_train_batch in gen_train_batch(X_train, X_train_meta, y_train, batch_size):
            
            X_meta_batch = X_meta_batch.reshape(-1,meta_dims)
            X_train_batch = scale(X_train_batch,mean_train,std_train)
            X_train_batch = X_train_batch.reshape(-1,frames,bins,chs)
            _,train_loss_batch = sess.run([train_op,loss],
               feed_dict={x:X_train_batch, meta_x:X_meta_batch, y:y_train_batch, is_training:True})
            train_loss += train_loss_batch ; n_batch += 1
        train_loss = train_loss/n_batch
        train_summary_op = tf.summary.merge([loss_summary])
        train_summaries = sess.run(train_summary_op,feed_dict={loss_all:train_loss})
        train_summary_writer.add_summary(train_summaries, epoch)
        
        print("step %d" %(epoch))
        print("   train loss: %f" % (train_loss))
        
        pre = []
        if ((epoch+1) % snapshot == 0 and epoch > 0) or epoch == n_epochs-1:
            sess.run(learning_rate_decay_op)
        
            for val_data_batch, val_meta_batch in gen_val_batch(X_valid, X_valid_meta, batch_size):
                  
                val_meta_batch = val_meta_batch.reshape(-1,meta_dims)
                val_data_batch = scale(val_data_batch,mean_train,std_train)
                val_data_batch = val_data_batch.reshape(-1,frames,bins,chs) 
                prediction = sess.run(sigmoid_output, feed_dict={x:val_data_batch, 
                                                         meta_x: val_meta_batch, is_training:False})
                pre.extend(prediction)
            # print(len(pre))
            generate_output_file(pre, valid_file_idxs, model_path, file_list, label_mode, taxonomy)
            submission_path = os.path.join(model_path, "output.csv")
            df_dict = metrics.evaluate(prediction_path=submission_path,annotation_path=annotation_path,
                                      yaml_path=taxonomy_path,mode=label_mode)  
            val_micro_auprc,eval_df = metrics.micro_averaged_auprc(df_dict,return_df=True) 
            val_macro_auprc,class_auprc = metrics.macro_averaged_auprc(df_dict,return_classwise=True)
            thresh_idx_05 = (eval_df['threshold']>=0.5).nonzero()[0][0]
            val_micro_F1score = eval_df['F'][thresh_idx_05]
    
            val_summaries = sess.run(val_summary_op,feed_dict={val_summary:val_micro_auprc})
            val_micro_auprc_summary_writer.add_summary(val_summaries, epoch)
            val_summaries = sess.run(val_summary_op,feed_dict={val_summary:val_macro_auprc})
            val_macro_auprc_summary_writer.add_summary(val_summaries, epoch)
            val_summaries = sess.run(val_summary_op,feed_dict={val_summary:val_micro_F1score})
            val_val_micro_F1score_summary_writer.add_summary(val_summaries, epoch)
            class_auprc_dict['class_auprc_'+str(epoch)] = class_auprc
            print('official')
            print('micro',val_micro_auprc)
            print('micro_F1',val_micro_F1score)
            print('macro',val_macro_auprc)
        
            print('-----save:{}-{}'.format(os.path.join(model_path,'ckeckpoint','model'), epoch))
            saver.save(sess, os.path.join(model_path,'ckeckpoint','model'), global_step=epoch)

    
            np.save(os.path.join(model_path,'class_auprc_dict.npy'),class_auprc_dict)
    sess.close()





if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", default='/annotations.csv')
    parser.add_argument("--taxonomy_path",  default='/dcase-ust-taxonomy.yaml')
    parser.add_argument("--output_dir", type=str,default='/models')
    parser.add_argument("--train_feature_dir", type=str, default='/logmel_h')
    parser.add_argument("--val_feature_dir", type=str, default='/logmel_h')
    parser.add_argument("--load_checkpoint_path", default='')
    
    parser.add_argument("--load_checkpoint", default=False)
    parser.add_argument("--chs", type=int, default=1)
    parser.add_argument("--kernel_size", type=int,default=3)
    parser.add_argument("--layer_depth", default=[64,128,256,256])
    parser.add_argument("--exp_id", type=str, default='1')
    parser.add_argument("--max_ckpt", type=int, default=20)
    parser.add_argument("--snapshot", type=int, default=5)

    parser.add_argument("--hidden_layer_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=50)
#    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='coarse')

    args = parser.parse_args()

    # save args to disk
    
    model_path = os.path.join(args.output_dir, 'exp'+args.exp_id)
    os.makedirs(model_path, exist_ok=True)
    kwarg_file = os.path.join(model_path, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(annotation_path = args.annotation_path, 
          taxonomy_path = args.taxonomy_path,  
          train_feature_dir = args.train_feature_dir, 
          val_feature_dir = args.val_feature_dir,
          output_dir = args.output_dir, 
          load_checkpoint = args.load_checkpoint,
          load_checkpoint_path = args.load_checkpoint_path,
          exp_id = args.exp_id, label_mode = "coarse", 
          batch_size = args.batch_size, n_epochs = args.n_epochs, 
          kernel_size = args.kernel_size, layer_depth = args.layer_depth, 
          chs = args.chs, max_ckpt = args.max_ckpt,
          lr = args.lr, hidden_layer_size = args.hidden_layer_size, 
          snapshot = args.snapshot,
          num_hidden_layers = args.num_hidden_layers, 
          standardize = True,
          timestamp = None)
