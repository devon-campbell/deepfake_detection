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
from utils import XvectorDataset, txt_to_df, get_protocol, load_xvectors, link_xvectors_and_labels, get_xvector_and_labels, get_dataloader
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)  # Apply residual connection
        x = self.layer_norm1(x)  # Apply layer normalization
        
        # Apply position-wise feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Apply residual connection
        x = self.layer_norm2(x)  # Apply layer normalization

        return x
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim=512, num_heads=4, num_layers=6, hidden_dim=256, output_dim=1):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)  # Input x-vectors are transformed here
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        x = self.final_layer(x)
        return self.sigmoid(x).squeeze()

class ASVDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_sample_rate=16000, fixed_length=160000, transform=None):  # fixed_length approx 10 seconds at 16kHz
        """
        Args:
            dataframe (DataFrame): Dataframe containing file paths and labels.
            target_sample_rate (int): Desired sample rate for all audio files.
            fixed_length (int): The fixed length of the audio signals (in samples).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.target_sample_rate = target_sample_rate
        self.fixed_length = fixed_length
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_path = self.dataframe.iloc[idx]['file_path']
        label = 1 if self.dataframe.iloc[idx]['key'] in {'bonafide', 'genuine'} else 0
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Ensure all audio files are the same length
        if waveform.size(1) > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]  # Trim to the fixed length
        elif waveform.size(1) < self.fixed_length:
            # Pad with zeros
            pad_amount = self.fixed_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
    
class VoxCelebDataset(Dataset):
    def __init__(self, xvectors, labels):
        """
        Args:
        xvectors (Tensor): Tensor of x-vectors.
        labels (list): List of labels (can be dummy since actual labels might not be necessary for pre-training).
        """
        self.xvectors = xvectors
        self.labels = labels

    def __len__(self):
        return len(self.xvectors)

    def __getitem__(self, idx):
        return self.xvectors[idx], self.labels[idx]
    
# class XvectorDataset(Dataset):
#     def __init__(self, dataframe):
#         """
#         Initialize the dataset with a DataFrame containing x-vectors and labels.

#         Args:
#         dataframe (DataFrame): DataFrame containing x-vectors and labels.
#         """
#         # Convert x-vector arrays in the DataFrame to tensors
#         self.xvectors = [torch.tensor(xvector, dtype=torch.float32) for xvector in dataframe['xvector_path']]
        
#         # Map labels 'bonafide' to 1 and 'spoof' to 0
#         self.labels = dataframe['key'].apply(lambda x: 1 if x == 'bonafide' else 0).tolist()

#     def __len__(self):
#         return len(self.xvectors)

#     def __getitem__(self, idx):
#         return self.xvectors[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

# def txt_to_df(file_path, cols):
#     with open(file_path, 'r') as file:
#         data = [line.strip().split() for line in file]
#     for item in data:
#         if '.wav' in item[0]:
#             item[0] = item[0][:-4]  # Remove '.wav'
#     return pd.DataFrame(data, columns=cols)

# def get_protocol(step, year):
#     if step == 'eval':
#         prot_path = 'asv2021/keys/DF/CM/trial_metadata.txt'
#         cols = ["speaker_id", "audio_file_name", "codec", "source_dataset", "source_team", "key", "trim_status",
#                 "eval_mode", "vocoder_type", "task", "team", "gender", "mode"]
#     else:
#         x = 'trl' if step == 'dev' else 'trn'
#         if year == 2017:
#             prot_path = f'asv2017/protocol_V2/ASVspoof2017_V2_{step}.{x}.txt'
#             cols = ["audio_file_name", "key", "speaker_id", "session_id", "environment_id", "play_id", "recording_id"]
#         elif year == 2019:
#             prot_path = f'asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{step}.{x}.txt'
#             cols = ["speaker_id", "audio_file_name", "environment_id", "attack_id", "key"]
#         else:
#             logging.error(f'Invalid data request: {step}, {year} is not available')
#             return None

#     full_path = os.path.expanduser(f'~/export/corpora/{prot_path}')
#     return txt_to_df(full_path, cols)
    
# def load_xvectors(directory_path):
#     """
#     Load VoxCeleb x-vectors from the specified directory.
    
#     Args:
#         directory_path (str): Path to the directory containing VoxCeleb .scp files.

#     Returns:
#         torch.Tensor: A tensor containing all x-vectors loaded from all files.
#     """
#     directory_path = os.path.expanduser(directory_path)  # Expands the ~ to the user's home directory path
#     if not os.path.exists(directory_path):
#         logging.warning(f"Directory does not exist: {directory_path}")
#         return torch.empty(0)

#     all_xvectors = []
#     for filename in os.listdir(directory_path):
#         try:
#             if filename.endswith('.scp'):
#                 scp_path = os.path.join(directory_path, filename)
#                 data = kaldiio.load_scp(scp_path)
#                 xvectors = [torch.tensor(vec, dtype=torch.float32) for vec in data.values()]
#                 all_xvectors.extend(xvectors)
#         except Exception as e:
#             logging.warning(f'Loading {filename} failed: {e}')

#     return torch.stack(all_xvectors) if all_xvectors else torch.empty(0)

# def link_xvectors_and_labels(directory_path, protocol_df):
#     """
#     Link x-vectors with their corresponding labels for all .scp files in a directory.

#     Args:
#     directory_path (str): Directory containing .scp files with x-vector references.
#     protocol_df (DataFrame): Loaded protocol DataFrame with labels.

#     Returns:
#     DataFrame: A DataFrame containing all x-vector paths and corresponding labels.
#     """
#     all_xvectors = []

#     # Iterate over all .scp files in the directory
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.scp'):
#             scp_path = os.path.join(directory_path, filename)
#             xvector_paths = kaldiio.load_scp(scp_path)
            
#             # Create a DataFrame from the x-vector paths
#             xvector_df = pd.DataFrame(list(xvector_paths.items()), columns=['audio_file_name', 'xvector_path'])
            
#             # Merge with protocol to link labels
#             merged_df = pd.merge(xvector_df, protocol_df, on='audio_file_name', how='left')
#             all_xvectors.append(merged_df)
    
#     # Concatenate all dataframes if there are any, else return empty dataframe
#     if all_xvectors:
#         return pd.concat(all_xvectors, ignore_index=True)
#     else:
#         return pd.DataFrame()

# def get_xvector_and_labels(step):
#     if step == 'eval':
#         return link_xvectors_and_labels(f'./asvspoof_xvector/data/eval/asvspoof2021/xvectors', get_protocol(step, 2021))
#     else:
#         df2017 = link_xvectors_and_labels(f'./asvspoof_xvector/data/{step}/asvspoof2017/xvectors', get_protocol(step, 2017))
#         df2019 = link_xvectors_and_labels(f'./asvspoof_xvector/data/{step}/asvspoof2019/xvectors', get_protocol(step, 2019))
#         return pd.concat([df2017, df2019], axis=0)

# def get_dataloader(step):
#     dataset = XvectorDataset(get_xvector_and_labels(step))
#     return DataLoader(dataset, batch_size=32, shuffle=True)

def pretrain_model(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xvector, labels in dataloader:
            xvector, labels = xvector.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(xvector).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logging.info(f"Pre-training Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

        # Save checkpoint after each epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
        }, filename=f"pretrain_checkpoint_epoch_{epoch+1}.pth.tar")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for xvector, labels in dataloader:
        xvector, labels = xvector.to(device), labels.to(device)
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(xvector)  # Forward pass
        loss = criterion(outputs.squeeze(), labels.float())  # Calculate the loss
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model weights
        
        total_loss += loss.item()
        predicted = outputs.round()  # Threshold the outputs to get binary predictions
        correct += (predicted.squeeze() == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def validate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss

def train_model(model, train_loader, dev_loader, criterion, optimizer, device, num_epochs):
    #best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()  # Ensure output matches target shape
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validate at the end of each epoch
        val_loss = validate_model(model, dev_loader, criterion, device)
        logging.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
        '''
        # Checkpointing logic
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
            }, filename="best_checkpoint.pth.tar")
        '''
        # Save regular checkpoints
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }, filename="train_checkpoint.pth.tar")

def collect_scores_and_labels(model, dataloader, device):
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for xvector, label in dataloader:
            xvector = xvector.to(device)
            output = model(xvector).squeeze()
            scores.extend(output.cpu().numpy())
            labels.extend(label.cpu().numpy())

    return scores, labels

def compute_fmr_fnmr(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnmr = 1 - tpr  # False non-match rate is 1 - true positive rate
    fmr = fpr  # False match rate is the false positive rate
    return fmr, fnmr, thresholds

def plot_det_curve(fmr, fnmr, epoch):
    plt.figure()
    axis_min = min(fmr.min(), fnmr.min()) * 100
    axis_max = max(fmr.max(), fnmr.max()) * 100

    plt.plot(norm.ppf(fmr), norm.ppf(fnmr), label='DET Curve')
    plt.xlabel('False Match Rate (%)')
    plt.ylabel('False Non-Match Rate (%)')
    plt.xlim(axis_min, axis_max)  # Use dynamic limits
    plt.ylim(axis_min, axis_max)  # Use dynamic limits
    plt.grid(True)
    plt.legend()
    plt.title('Detection Error Tradeoff Curve')
    plt.savefig(f'./graphs/det_{epoch}.png')
    
def evaluate_and_plot_det(model, dataloader, device, epoch):
    scores, labels = collect_scores_and_labels(model, dataloader, device)
    fmr, fnmr, thresholds = compute_fmr_fnmr(np.array(labels), np.array(scores))
    eer_index = np.nanargmin(np.abs(fmr - fnmr))
    eer = (fmr[eer_index] + fnmr[eer_index]) / 2
    plot_det_curve(fmr, fnmr, epoch)
    return eer, thresholds[eer_index]

def save_checkpoint(state, filename="checkpoint.pth.tar", path="checkpoints"):
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    logging.info(f"Checkpoint saved to {full_path}")

def load_checkpoint(filename, model, optimizer, path="checkpoints"):
    full_path = os.path.join(path, filename)
    checkpoint = torch.load(full_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    logging.info(f"Loaded checkpoint '{full_path}' (epoch {epoch})")
    return model, optimizer, epoch

def pretrain_voxceleb(model, criterion, optimizer, device, epochs=5):
    # Setup the pre-training environment
    xvector_dir_path = os.path.expanduser('~/kaldi-trunk/egs/voxceleb/v2/exp/xvector_nnet_1a/xvectors_train')
    voxceleb_xvectors = load_xvectors(xvector_dir_path)
    voxceleb_labels = torch.ones(voxceleb_xvectors.size(0))  # Label as 1 for bonafide
    voxceleb_dataset = VoxCelebDataset(voxceleb_xvectors, voxceleb_labels)
    voxceleb_loader = DataLoader(voxceleb_dataset, batch_size=32, shuffle=True)

    logging.info("Loaded VoxCeleb xvectors")

    # Pre-training model on VoxCeleb xvectors
    logging.info("Begin pre-training on VoxCeleb xvectors")
    pretrain_model(model, voxceleb_loader, criterion, optimizer, device, epochs)
    logging.info("Completed pre-training on VoxCeleb xvectors")

def check_pretraining_completion(epochs, path="checkpoints"):
    # Check if the last epoch's checkpoint exists
    last_epoch_checkpoint = os.path.join(path, f"pretrain_checkpoint_epoch_{epochs}.pth.tar")
    return os.path.exists(last_epoch_checkpoint)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_dim=512, num_heads=4, num_layers=6, hidden_dim=256, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    os.makedirs('graphs', exist_ok=True)  # For DET curve graphs
    os.makedirs('checkpoints', exist_ok=True)  # For model checkpoints

    pretraining_epochs = 5
    pretrain_checkpoint = f"pretrain_checkpoint_epoch_{pretraining_epochs}.pth.tar"
    pretrain_checkpoint_path = os.path.join("checkpoints", pretrain_checkpoint)
    if os.path.exists(pretrain_checkpoint_path):
        logging.info("Pretraining already completed, skipping...")
        model, optimizer, _ = load_checkpoint(pretrain_checkpoint, model, optimizer)
        logging.info("Loaded pretraining checkpoint.")
    else:
        logging.info("Starting pretraining")
        pretrain_voxceleb(model, criterion, optimizer, device, pretraining_epochs)

    #main_checkpoint_path = os.path.join('checkpoints', 'train_checkpoint.pth.tar')
    main_checkpoint ='train_checkpoint.pth.tar'
    if os.path.isfile(os.path.join('checkpoints', main_checkpoint)):
        model, optimizer, start_epoch = load_checkpoint(main_checkpoint, model, optimizer)
        logging.info(f"Resuming main training from epoch {start_epoch}")
    else:
        start_epoch = 0
        logging.info("Starting fresh main training")

    train_loader, dev_loader, eval_loader = get_dataloader('train'), get_dataloader('dev'), get_dataloader('eval')
    logging.info("Retrieved train & eval dataloaders")

    #Main training loop
    num_epochs = 10
    train_model(model, train_loader, dev_loader, criterion, optimizer, device, num_epochs - start_epoch)  

    # Evaluation  
    eer, threshold = evaluate_and_plot_det(model, eval_loader, device, num_epochs)
    logging.info(f"End of Evaluation: EER = {eer:.2f}, Threshold = {threshold:.4f}")

if __name__ == "__main__":
    main()