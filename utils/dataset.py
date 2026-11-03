
import librosa
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

audio_path = 'ESC-50/audio'
mel_path = 'ESC-50/melSpec'
meta_path  = 'ESC-50/meta/esc50.csv'

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

class ESC50_Dataset(Dataset):
  def __init__(self, meta_data, path_to_file, path_to_store,
               sample_rate=44100, n_mels=64, target_length=512, 
               rewrite=False, skip_extract=False) -> None:
    super().__init__()

    self.dataset = meta_data
    self.path_to_store = path_to_store

    def extractData(row):
      file_name = row['filename']

      file_path = path_to_file + '/' + file_name

      # Create saving folder
      create_folder(path_to_store)
      # Extract the filename from the extension
      f_name, extension = os.path.splitext(file_name)
      file_store_path = path_to_store + '/' + f_name + '.npy'
      if os.path.exists(file_store_path) and not rewrite:
        return

      y, sr = librosa.load(file_path, sr=sample_rate)

      mel_spec = librosa.feature.melspectrogram(
          y=y,
          sr=sr,
          n_mels=n_mels,     # Determine the number of frequency bins
          n_fft=2048,
          hop_length=430
      )

      # Convert power spectrogram to logarithmic scale (Decibels)
      mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

      current_width = mel_spec_db.shape[1]
      if current_width > target_length:
        mel_spec_db = mel_spec_db[:, : target_length]
      else:
        pad_width = target_length - current_width
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)),
                             mode='constant')

      np.save(file_path, mel_spec_db)

    if not skip_extract:
      self.dataset.apply(extractData, axis=1)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    row = self.dataset.iloc[idx]
    filename = row['filename']
    target = row['target']

    f_name, extension = os.path.splitext(filename)

    mel_spec_db = np.load(self.path_to_store + '/' + f_name + '.npy')

    return mel_spec_db, target

def get_validation_dataset(path):
  df = pd.read_csv(path)
  return df[df['fold'] == 4]

def get_training_dataset(path):
  df = pd.read_csv(path)
  return df[(df['fold'] == 1) | (df['fold'] == 2) | (df['fold'] == 3)]

def get_test_dataset(path):
  df = pd.read_csv(path)
  return df[df['fold'] == 5]