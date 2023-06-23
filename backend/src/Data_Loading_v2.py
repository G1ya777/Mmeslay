from torch.utils.data import Dataset,DataLoader
from torchaudio.transforms import MelSpectrogram, FrequencyMasking,TimeMasking
from torch.nn.utils.rnn import pad_sequence
from torch import tensor,int16,from_numpy,log
import pandas as pd
import sentencepiece as spm
import random
import numpy as np
from scipy.io.wavfile import read
from torchaudio.transforms import Resample
from torchaudio import info,load
from audiomentations import Compose, TimeStretch as TimeStretch_waveform, LowPassFilter, Mp3Compression, AddGaussianNoise


# from SpecAugment import spec_augment_pytorch

sp = spm.SentencePieceProcessor()
sp.load("./ressources/tokenizer/128_v7.model")



train = pd.read_csv("./ressources/train.csv",
engine='c',
low_memory=False,)


val = pd.read_csv("./ressources/dev_s2.csv",
engine='c',
low_memory=False,)
#[:4350].reset_index()
test = pd.read_csv("./ressources/test_s2.csv",
engine='c',
low_memory=False,)


# X_train = train["path, path"]
X_train = train["path"]
y_train = train["sentence"]
X_val = val["path"]
y_val = val["sentence"]
X_test = test["path"]
y_test = test["sentence"]
del test
del train,val


def collate_fn (batch):
    if None not in batch:
        transcriptions,spectrograms,audio_lengths = zip(*batch)
        transcriptions_lengths = tensor([transcription.shape[0] for transcription in transcriptions],dtype=int16)
        specs_lengths = tensor([spec.shape[0] for spec in spectrograms],dtype=int16)
        padded_spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
        padded_transcriptions = pad_sequence(transcriptions, batch_first=True, padding_value=0)

        return padded_spectrograms, padded_transcriptions,tensor(audio_lengths),transcriptions_lengths
    return None 


class AudioDataset(Dataset):

    def __init__(self, X,y,train=False,val=False,):

        self.audio_dirs = X
        self.transcription = y
        self.train = train
        self.val = val
        

    def __len__(self):

        return len(self.transcription)

    def __getitem__(self, idx):
        
        dir = str(self.audio_dirs[idx])
        dir = dir.split(",")
        if self.train:
            id = random.randint(0, len(dir)-1)
        else:
            id = 0
        # id = 0
        
        
        audio_dir = "/mount/ADATA_HV300/clips_2/"+dir[id]+".mp3"

        transcription = tensor(sp.EncodeAsIds(self.transcription[idx]))
        waveform,sr = load(audio_dir,normalize=True)
        audio_info = info(audio_dir)
        # audio_length = audio_info.num_frames / audio_info.sample_rate
        if audio_info.num_channels == 2 :
                waveform = waveform[1:2, :]
        waveform = waveform.squeeze()
        transform = Resample(sr, 16000)
        sr = 16000
        waveform = transform(waveform)


        return transcription, waveform, len(waveform)




train_data = AudioDataset(X_train,y_train,train=True)
validation_data = AudioDataset(X_val,y_val,val=True)
# train_dataloader = DataLoader(train_data,shuffle=True, batch_size=16,num_worker8s=0,collate_fn=collate_fn)
train_dataloader = DataLoader(train_data,shuffle=True,drop_last=True,batch_size=64,num_workers=8, collate_fn=collate_fn,pin_memory=True,persistent_workers=True)
validation_dataloader = DataLoader(validation_data, batch_size=64,num_workers=4, collate_fn=collate_fn, persistent_workers=True)

test_data = AudioDataset(X_test,y_test)
test_dataloader = DataLoader(test_data, batch_size=4,num_workers=4, collate_fn=collate_fn)


# import multiprocessing as mp

# if __name__ == '__main__':
#     mp.set_start_method('spawn')
# for batch_idx, batch_data in enumerate(train_dataloader):
#     # print()
#     # break
#     pass