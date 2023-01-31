import os
import shutil
import torch
import scipy.io as scio
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import Wav2Vec2FeatureExtractor
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataloader import wavDataset
import pickle


label_list = ['negative', 'positive']
class model_args:
    config_name = "pretrained/config.json"

class data_args:
    input_column = "path"
    target_column = "class"

    # 載入資料集
    source_train = 'audioset/source_train.csv'
    source_valid = 'audioset/source_valid.csv'
    target_train = 'audioset/target_train.csv'
    target_valid = 'testing/CutAudio_BWF_2018_1hr_4sec.csv'

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    model_args.config_name,
    revision="main", 
    use_auth_token=False
)
target_sampling_rate = feature_extractor.sampling_rate
input_column_name = data_args.input_column
output_column_name = data_args.target_column

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def preprocess_function(df):
    speech_list = [speech_file_to_array_fn(path) for path in df[input_column_name]]
    target_list = [label_to_id(label, label_list) for label in df[output_column_name]]
    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    
    #! modify
    for i in range(len(result["input_values"])):
        arr = result["input_values"][i]
        arr[np.isnan(arr)] = 0
        dim = arr.shape
        if len(dim) > 1:
            result["input_values"][i] = np.mean(arr, axis=0)

    return result


def generate_dataloader(args):
    # Data loading code
    source_train = pd.read_csv(data_args.source_train)
    source_valid = pd.read_csv(data_args.source_valid)
    target_train = pd.read_csv(data_args.target_train)
    target_valid = pd.read_csv(data_args.target_valid)

    source_train = preprocess_function(source_train)
    source_valid = preprocess_function(source_valid)
    target_train = preprocess_function(target_train)
    target_valid = preprocess_function(target_valid)

    source_train = wavDataset(source_train, args)
    source_valid = wavDataset(source_valid, args)
    target_train = wavDataset(target_train, args)
    target_valid = wavDataset(target_valid, args)
    
    # PadCollate的功能在於，將統一同batch中長短不一的音訊，短的一方會被pad 0直到與最長的那方相同
    source_train_loader = DataLoader(dataset=source_train, batch_size=args.batch_size, shuffle=True, collate_fn=PadCollate(dim=0))
    source_valid_loader = DataLoader(dataset=source_valid, batch_size=args.batch_size, shuffle=False, collate_fn=PadCollate(dim=0))
    target_train_loader = DataLoader(dataset=target_train, batch_size=args.batch_size, shuffle=True, collate_fn=PadCollate(dim=0))
    target_valid_loader = DataLoader(dataset=target_valid, batch_size=args.batch_size, shuffle=False, collate_fn=PadCollate(dim=0))

    return source_train_loader, target_train_loader, source_valid_loader, target_valid_loader


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)

    vec = vec.cpu()
    tmp = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

    return tmp


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))

        # pad according to max_len
        batch = list(map(lambda x: (pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]), batch))
        
        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)