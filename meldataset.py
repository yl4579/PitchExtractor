#coding: utf-8
"""
TODO:
- make TestDataset
- separate transforms
"""

import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import pyworld as pw

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 verbose=True
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [d[0] for d in _data_list]

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.mean, self.std = -4, 4
        
        self.verbose = verbose
        
        # for silence detection
        self.zero_value = -10 # what the zero value is
        self.bad_F0 = 5 # if less than 5 frames are non-zero, it's a bad F0, try another algorithm

    def __len__(self):
        return len(self.data_list)

    def path_to_mel_and_label(self, path):
        wave_tensor = self._load_tensor(path)
        
        # use pyworld to get F0
        output_file = path + "_f0.npy"
        # check if the file exists
        if os.path.isfile(output_file): # if exists, load it directly
            f0 = np.load(output_file)
        else: # if not exist, create F0 file
            if self.verbose:
                print('Computing F0 for ' + path + '...')
            x = wave_tensor.numpy().astype("double")
            frame_period = MEL_PARAMS['hop_length'] * 1000 / self.sr
            _f0, t = pw.harvest(x, self.sr, frame_period=frame_period)
            if sum(_f0 != 0) < self.bad_F0: # this happens when the algorithm fails
                _f0, t = pw.dio(x, self.sr, frame_period=frame_period) # if harvest fails, try dio
            f0 = pw.stonemask(x, _f0, t, self.sr)
            # save the f0 info for later use
            np.save(output_file, f0)
        
        f0 = torch.from_numpy(f0).float()
        
        if self.data_augmentation:
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        
        f0_zero = (f0 == 0)
        
        #######################################
        # You may want your own silence labels here
        # The more accurate the label, the better the resultss
        is_silence = torch.zeros(f0.shape)
        is_silence[f0_zero] = 1
        #######################################
        
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
            f0 = f0[random_start:random_start + self.max_mel_length]
            is_silence = is_silence[random_start:random_start + self.max_mel_length]
        
        if torch.any(torch.isnan(f0)): # failed
            f0[torch.isnan(f0)] = self.zero_value # replace nan value with 0
        
        return mel_tensor, f0, is_silence


    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, f0, is_silence = self.path_to_mel_and_label(data)
        return mel_tensor, f0, is_silence

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        f0s = torch.zeros((batch_size, self.max_mel_length)).float()
        is_silences = torch.zeros((batch_size, self.max_mel_length)).float()

        for bid, (mel, f0, is_silence) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            f0s[bid, :mel_size] = f0
            is_silences[bid, :mel_size] = is_silence

        if self.max_mel_length > self.min_mel_length:
            random_slice = np.random.randint(
                self.min_mel_length//self.mel_length_step,
                1+self.max_mel_length//self.mel_length_step) * self.mel_length_step + self.min_mel_length
            mels = mels[:, :, :random_slice]
            f0 = f0[:, :random_slice]

        mels = mels.unsqueeze(1)
        return mels, f0s, is_silences


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
