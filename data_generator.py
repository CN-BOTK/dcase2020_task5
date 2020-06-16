#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:21:15 2019

@author: dcase
"""
import numpy as np
from dcase_functions import *
import os 
import librosa
import pandas as pd
import math
import scipy.sparse as ss
import yaml
import random

def get_train_audiodata(train_label_csv,audio_path):
    name_list = list(train_label_csv['audio_filename'])
    data = []
    label = np.asarray(train_label_csv)
    label = list(np.asarray(label[:,1:],dtype=np.float32))
    for name in name_list:
        file_path =os.path.join(audio_path,name)
        y,sr = librosa.load(file_path,sr = 22050)
        if len(y)!=sr*10:
            y = np.resize(y,sr*10)
        data.append(y)
    return np.asarray(data,np.float32),np.asarray(label,np.float32)

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
        # if split != "train":
        if split != "train" and ann_id < 0:
            continue

        for label in labels:
            count_dict[fname][label] += row[label + '_presence']

    targets = np.array([[1.0 if count_dict[fname][label] > 0 else 0.0 for label in labels]
                        for fname in file_list])

    return targets

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
    test_idxs = []
    test_val_idxs = []
    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        # elif row['split'] == 'validate' and row['annotator_id'] <= 0:
        elif row['split'] == 'validate' :
            # For validation examples, only use verified annotations
            valid_idxs.append(idx)
            if row['annotator_id'] <= 0 :
                test_val_idxs.append(idx)
        elif row['split'] == 'test' :
            test_idxs.append(idx)
    return np.array(train_idxs), np.array(valid_idxs), np.array(test_idxs), np.array(test_val_idxs)
def one_hot(idx, num_items):
    return [(0.0 if n != idx else 1.0) for n in range(num_items)]

NUM_HOURS = 24
NUM_DAYS = 7
NUM_WEEKS = 52
def prepare_data(train_file_idxs, valid_file_idxs,
                 latitude_list, longitude_list, week_list, day_list, hour_list,
                 target_list, standardize=True):

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

def get_feature(audio,feature):
#    train_data=[]
    if feature == 'log_mel':  
#        for audio in train_audio_data:
            data = gen_mel_features(audio)            
#            train_data.append(data)
    if feature == 'STFT':
#        for audio in train_audio_data:
            data = STFT(audio)
            data = power_to_dB(np.abs(data)**2).T
#            train_data.append(data)
    if feature == 'HPSS_h':        
#        for audio in train_audio_data:
            data = STFT(audio)
            data_h,data_p = librosa.decompose.hpss(np.abs(data)**2)
            data = power_to_dB(data_h).T
#            train_data.append(data_h)
    if feature == 'HPSS_p':        
#        for audio in train_audio_data:
            data = STFT(audio)
            data_h,data_p = librosa.decompose.hpss(np.abs(data)**2)
            data = power_to_dB(data_p).T
    if feature == 'log_linear':   
#        for audio in train_audio_data:
            data = gen_linear_features(audio,64)            
#            train_data.append(data)
#    return np.asarray(train_data,np.float32)
    if feature == 'mfcc':
        data = librosa.feature.mfcc(y=audio,sr=22050,n_fft=1024,hop_length=512,n_mfcc=40).T
    if feature == 'log_mel_h':
        data = gen_mel_h_features(audio)
    if feature == 'log_mel_p':
        data = gen_mel_p_features(audio)
        
    return data

# def get_data(label_csv,audio_path):
#     name_list = list(label_csv['audio_filename'])
#     data = []
#     for name in name_list:
#         file_path =os.path.join(audio_path,name)
#         y,sr = librosa.load(file_path,sr = 32000)
#         if len(y)!=sr*10:
#             y = np.concatenate((y, np.zeros(sr*10 - len(y))))
#         data.append(y)
#     return np.asarray(data,np.float32)

def gen_mel_features(data):
#    train_data = []
#    for i in range(data.shape[0]):        
    stft_matric = librosa.core.stft(data,n_fft=1024,hop_length=512,win_length=1024,window='hann')### 513*frames
    mel_W = librosa.filters.mel(sr=22050,n_fft=1024,n_mels=64,fmin=50,fmax=10000)       ### 64*frames
    mel_spec = np.dot(mel_W,(np.abs(stft_matric))**2)          
    log_mel = librosa.core.power_to_db(mel_spec,top_db=None).T  ###   frames*64
#    train_data.append(log_mel)
    return log_mel

def gen_mel_h_features(data):
      
    stft_matric = librosa.core.stft(data,n_fft=1024,hop_length=512,win_length=1024,window='hann')### 513*frames
    mel_W = librosa.filters.mel(sr=22050,n_fft=1024,n_mels=64,fmin=50,fmax=10000)       ### 64*frames
    data_h,data_p = librosa.decompose.hpss(np.abs(stft_matric)**2)
    mel_h_spec = np.dot(mel_W,data_h)          
    log_mel_h = librosa.core.power_to_db(mel_h_spec,top_db=None).T  ###   frames*64

    return log_mel_h

def gen_mel_p_features(data):
      
    stft_matric = librosa.core.stft(data,n_fft=1024,hop_length=512,win_length=1024,window='hann')### 513*frames
    mel_W = librosa.filters.mel(sr=22050,n_fft=1024,n_mels=64,fmin=50,fmax=10000)       ### 64*frames
    data_h,data_p = librosa.decompose.hpss(np.abs(stft_matric)**2)
    mel_p_spec = np.dot(mel_W,data_p)          
    log_mel_p = librosa.core.power_to_db(mel_p_spec,top_db=None).T  ###   frames*64

    return log_mel_p

def linear_frequency(fs, NbCh, nfft, warp, fhigh, flow):
    warp = 1
    fhigh = fs / 2
    flow = 0
    LowMel = flow
    NyqMel = fhigh

    StartMel = LowMel + np.linspace(0, NbCh - 1, NbCh) / (NbCh + 1) * (NyqMel - LowMel)
    fCen = StartMel
    StartBin = np.round(nfft / fs * fCen) + 1

    EndMel = LowMel + np.linspace(2, NbCh + 1, NbCh) / (NbCh + 1) * (NyqMel - LowMel)
    EndBin = np.round(warp * nfft / fs * EndMel)

    TotLen = EndBin - StartBin + 1

    LowLen = np.append(StartBin[1:NbCh], EndBin[NbCh - 2]) - StartBin + 1
    HiLen = TotLen - LowLen + 1

    M = ss.lil_matrix((math.ceil(warp * nfft / 2 + 1), NbCh)).toarray()
    for i in range(NbCh):
        # print(M[int(StartBin[i]-1):int(StartBin[i] + LowLen[i] - 1), i].shape)
        # print(((np.linspace(1, LowLen[i], LowLen[i])).T / LowLen[i]).reshape(-1,1).shape)
#        M[int(StartBin[i]-1):int(StartBin[i] + LowLen[i] - 1), i] = ((np.linspace(1, LowLen[i], LowLen[i])).T / LowLen[i])
#        M[int(EndBin[i] - HiLen[i]):int(EndBin[i]), i] = ((np.linspace(1, HiLen[i], HiLen[i])[::-1]).T / HiLen[i])
        M[int(StartBin[i]-1):int(StartBin[i] + LowLen[i] - 1), i] = ((np.linspace(1, int(LowLen[i]), int(LowLen[i]))).T / LowLen[i])
        M[int(EndBin[i] - HiLen[i]):int(EndBin[i]), i] = ((np.linspace(1, int(HiLen[i]), int(HiLen[i]))[::-1]).T / HiLen[i])
    Mfull = M
    M = M[0:int(nfft / 2+1),:]
    return M, Mfull

def gen_linear_features(data,bands_num):
#    train_data = []
#    for i in range(data.shape[0]):
    stft_matric = librosa.core.stft(data,n_fft=1024,hop_length=512,win_length=1024,window='hann')
    linear_W , _ = linear_frequency(fs=22050,NbCh=bands_num,nfft=1024,warp=1,fhigh=10000,flow=50)
    linear_spec = np.dot(linear_W.T,np.abs(stft_matric)**2)
    log_linear = librosa.core.power_to_db(linear_spec,top_db=None).T
#    train_data.append(log_linear)
    return log_linear

def train_data_augmentation(aug_loop_nums,aug_name_dict,root_audio_path,aug_path):
    aug_file_label_dict = {}
    aug_class = [1,2,3,5,7]
    for loop in range(aug_loop_nums):
        random_class_num = np.random.randint(2,6)
        random_class = random.sample(aug_class,random_class_num)
        aug_wav_label = np.zeros(8)
        for i in random_class:
            random_class_id = random_class[i]
            aug_wav_label[random_class_id]+=1
            if random_class[i]==1:
                random_amp_factor = random.uniform(0.7,1)
            elif random_class[i]==2:
                random_amp_factor = random.uniform(1,1.4)
            elif random_class[i]==3:
                random_amp_factor = random.uniform(0.2,0.4)
            elif random_class[i]==5:
                random_amp_factor = random.uniform(1.5,1.8)
            random_class_file_list = aug_name_dict[random_class_id]
            random_class_file_name = random.sample(random_class_file_list,1)
            random_class_file_path = os.path.join(root_audio_path,random_class_file_name)
            wav,sr = librosa.load(random_class_file_path,sr=22050)
            if i==0:
                aug_wav = random_amp_factor*wav
            else:
                aug_wav+= random_amp_factor*wav
        aug_wav_name = 'aug_wav_'+str(loop)+'.wav'
        output_path = os.path_join(aug_path,aug_wav_name)
        librosa.output.write_wav(output_path, aug_wav, sr)
        aug_file_label_dict[aug_wav_name] = aug_wav_label
    return aug_file_label_dict

if __name__=="__main__":
    
    
    annotation_path = '/home/dcase/c2020/dcase2020_task5/annotations.csv'
    taxonomy_path = '/home/dcase/c2020/dcase2020_task5/dcase-ust-taxonomy.yaml'
    audios_path = '/home/dcase/c2020/dcase2020_task5/audio'
    test_audios_path = '/home/dcase/c2020/dcase2020_task5/audio-eval'
    train_feature_path = '/home/dcase/c2020/BJS/2020dcase/task5/train/train_data/logmel_h'
    # if not os.path.exists(train_feature_path):
    #     os.mkdir(train_feature_path)
    # val_feature_path = '/home/dcase/c2020/BJS/2020dcase/task5/train/val_data/logmel_h'
    # if not os.path.exists(val_feature_path):
    #     os.mkdir(val_feature_path)
    test_feature_path = '/home/dcase/c2020/BJS/2020dcase/task5/test/test_data/log_mel_h'
    if not os.path.exists(test_feature_path):
        os.mkdir(test_feature_path)
    # train_one_class_dict_path = '/home/dcase/c2020/BJS/2020dcase/task5/label_csvs/train_coarse_one_class_dict.npy'
    
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)
#
    annotation_data_trunc = annotation_data[['audio_filename',
                                              'latitude',
                                              'longitude',
                                              'week',
                                              'day',
                                              'hour']].drop_duplicates()
    file_list = annotation_data_trunc['audio_filename'].to_list()
    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                                for coarse_id, fine_dict in taxonomy['fine'].items()
                                for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                          if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, valid_file_idxs, test_file_idxs,  test_val_file_idxs = get_subset_split(annotation_data)
    
    # train_one_class_dict = np.load(train_one_class_dict_path)
    
# train data gen
    # print('gen train data')
    # for train_id in train_file_idxs:
        
    #     audio_path = os.path.join(audios_path,file_list[train_id])
    #     audio, sr = librosa.load(audio_path,sr = 22050)
    #     train_data = get_feature(audio,feature='log_mel_h')
    #     np.save(os.path.join(train_feature_path,file_list[train_id][:-4]+'.npy'),train_data)
        
    # print('gen valid data')

    # val_feature_list = os.listdir(val_feature_path)
    # for val_id in valid_file_idxs:
    #     if file_list[val_id][:-4]+'.npy' in val_feature_list:
    #         continue
    #     audio_path = os.path.join(audios_path,file_list[val_id])
    #     audio, sr = librosa.load(audio_path,sr = 22050)
    #     val_data = get_feature(audio,feature='log_mel_h')
    #     np.save(os.path.join(val_feature_path,file_list[val_id][:-4]+'.npy'),val_data)
        
        
# test data gen
    print('gen test data')
    for test_id in test_file_idxs:       
        audio_path = os.path.join(test_audios_path,file_list[test_id])
        audio, sr = librosa.load(audio_path,sr = 22050)
        test_data = get_feature(audio,feature='log_mel_h')
        np.save(os.path.join(test_feature_path,file_list[test_id][:-4]+'.npy'),test_data)


## aug wavs gen
    # aug_name_dict_path = '/home/dcase/c2020/BJS/2020dcase/task5/label_csvs/aug_name_dict.npy'
    # aug_loop_nums = 5000
    # aug_name_dict = np.load(aug_name_dict_path).item()
    # root_audio_path = '/home/dcase/c2020/dcase2020_task5/audio'
    # aug_path = '/home/dcase/c2020/BJS/2020dcase/task5/train/aug_wavs'
    
    # aug_file_label_dict = {}
    # aug_class = [1,2,3,5,7]
    # for loop in range(aug_loop_nums):
    #     random_class_num = np.random.randint(2,6)
    #     random_class = random.sample(aug_class,random_class_num)
    #     aug_wav_label = np.zeros(8)
    #     for i in range(len(random_class)):
    #         random_class_id = random_class[i]
    #         aug_wav_label[random_class_id]+=1
    #         if random_class[i]==1:
    #             random_amp_factor = random.uniform(0.1,0.15)
    #         elif random_class[i]==2:
    #             random_amp_factor = random.uniform(1.1,1.3)
    #         elif random_class[i]==3:
    #             random_amp_factor = random.uniform(0.1,0.15)
    #         elif random_class[i]==5:
    #             random_amp_factor = random.uniform(1.9,2.3)
    #         elif random_class[i]==7:
    #             random_amp_factor = random.uniform(1,1.2)
    #         random_class_file_list = aug_name_dict[random_class_id]
    #         random_class_file_name = random.sample(random_class_file_list,1)
    #         random_class_file_path = os.path.join(root_audio_path,random_class_file_name[0])
    #         wav,sr = librosa.load(random_class_file_path,sr=22050)
    #         if i==0:
    #             aug_wav = random_amp_factor*wav
    #         else:
    #             aug_wav+= random_amp_factor*wav
    #     aug_wav_name = 'aug_wav_'+str(loop)+'.wav'
    #     output_path = os.path.join(aug_path,aug_wav_name)
    #     librosa.output.write_wav(output_path, aug_wav, sr)
    #     aug_file_label_dict[aug_wav_name] = aug_wav_label
    
    
    # aug train_data gen
    # aug_fea_path = '/home/dcase/c2020/BJS/2020dcase/task5/train/aug_train_data/logmel_64'
    # aug_wavs_path = '/home/dcase/c2020/BJS/2020dcase/task5/train/aug_train_wavs'
    # aug_wavs = os.listdir(aug_wavs_path)
    # for aug_wav_name in aug_wavs:
        
    #     aug_wav_path = os.path.join(aug_wavs_path,aug_wav_name)
    #     aug_wav, sr = librosa.load(aug_wav_path,sr = 22050)
    #     aug_data = get_feature(aug_wav,feature='log_mel')
    #     np.save(os.path.join(aug_fea_path,aug_wav_name[:-4]+'.npy'),aug_data)
