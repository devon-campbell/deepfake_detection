import os
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import kaldiio
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy.stats import norm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class XvectorDataset(Dataset):
    def __init__(self, dataframe):
        self.xvectors = torch.stack([torch.tensor(xvector, dtype=torch.float32) for xvector in dataframe['xvector_path']])
        self.labels = torch.tensor(dataframe['key'].apply(lambda x: 1 if x in {'bonafide','genuine'} else 0).tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.xvectors)

    def __getitem__(self, idx):
        return self.xvectors[idx], self.labels[idx]

def txt_to_df(file_path, cols):
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file]
    for item in data:
        if '.wav' in item[0]:
            item[0] = item[0][:-4]  # Remove '.wav'
    return pd.DataFrame(data, columns=cols)

def get_protocol(step, year):
    if step == 'eval':
        prot_path = 'asv2021/keys/DF/CM/trial_metadata.txt'
        cols = ["speaker_id", "audio_file_name", "codec", "source_dataset", "source_team", "key", "trim_status",
                "eval_mode", "vocoder_type", "task", "team", "gender", "mode"]
    else:
        x = 'trl' if step == 'dev' else 'trn'
        if year == 2017:
            prot_path = f'asv2017/protocol_V2/ASVspoof2017_V2_{step}.{x}.txt'
            cols = ["audio_file_name", "key", "speaker_id", "session_id", "environment_id", "play_id", "recording_id"]
        elif year == 2019:
            prot_path = f'asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{step}.{x}.txt'
            cols = ["speaker_id", "audio_file_name", "environment_id", "attack_id", "key"]
        else:
            logging.error(f'Invalid data request: {step}, {year} is not available')
            return None

    full_path = os.path.expanduser(f'./export/corpora/{prot_path}')
    return txt_to_df(full_path, cols)
    
def load_xvectors(directory_path):
    """
    Load VoxCeleb x-vectors from the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing VoxCeleb .scp files.

    Returns:
        torch.Tensor: A tensor containing all x-vectors loaded from all files.
    """
    directory_path = os.path.expanduser(directory_path)  # Expands the ~ to the user's home directory path
    if not os.path.exists(directory_path):
        logging.warning(f"Directory does not exist: {directory_path}")
        return torch.empty(0)

    all_xvectors = []
    for filename in os.listdir(directory_path):
        try:
            if filename.endswith('.scp'):
                scp_path = os.path.join(directory_path, filename)
                data = kaldiio.load_scp(scp_path)
                xvectors = [torch.tensor(vec, dtype=torch.float32) for vec in data.values()]
                all_xvectors.extend(xvectors)
        except Exception as e:
            logging.warning(f'Loading {filename} failed: {e}')

    return torch.stack(all_xvectors) if all_xvectors else torch.empty(0)

def link_xvectors_and_labels(directory_path, protocol_df):
    """
    Link x-vectors with their corresponding labels for all .scp files in a directory.

    Args:
    directory_path (str): Directory containing .scp files with x-vector references.
    protocol_df (DataFrame): Loaded protocol DataFrame with labels.

    Returns:
    DataFrame: A DataFrame containing all x-vector paths and corresponding labels.
    """
    all_xvectors = []

    # Iterate over all .scp files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.scp'):
            scp_path = os.path.join(directory_path, filename)
            xvector_paths = kaldiio.load_scp(scp_path)
            
            # Create a DataFrame from the x-vector paths
            xvector_df = pd.DataFrame(list(xvector_paths.items()), columns=['audio_file_name', 'xvector_path'])
            
            # Merge with protocol to link labels
            merged_df = pd.merge(xvector_df, protocol_df, on='audio_file_name', how='left')
            all_xvectors.append(merged_df)
    
    # Concatenate all dataframes if there are any, else return empty dataframe
    if all_xvectors:
        return pd.concat(all_xvectors, ignore_index=True)
    else:
        return pd.DataFrame()

def get_xvector_and_labels(step):
    if step == 'eval':
        return link_xvectors_and_labels(f'./asvspoof_xvector/data/eval/asvspoof2021/xvectors', get_protocol(step, 2021))
    else:
        df2017 = link_xvectors_and_labels(f'./asvspoof_xvector/data/{step}/asvspoof2017/xvectors', get_protocol(step, 2017))
        df2019 = link_xvectors_and_labels(f'./asvspoof_xvector/data/{step}/asvspoof2019/xvectors', get_protocol(step, 2019))
        return pd.concat([df2017, df2019], axis=0)

def get_dataloader(step):
    dataset = XvectorDataset(get_xvector_and_labels(step))
    return DataLoader(dataset, batch_size=32, shuffle=True)