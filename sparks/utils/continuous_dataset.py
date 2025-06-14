import torch
from torch.utils.data import Dataset
import numpy as np

class ContinuousChunkDataset(Dataset):
    """
    Dataset for continuous neural data that creates chunks from specified indices
    while filtering out chunks with insufficient labeled samples.
    """
    
    def __init__(self, X, y, chunk_length, tau_p, tau_f, stride=None, min_label_fraction=0.1):
        """
        Args:
            X: Neural data tensor [T, N_neurons] 
            y: Target labels tensor [T, N_labels]
            chunk_length: Length of each data chunk for training
            stride: Step size between chunks (default: chunk_length // 2)
            min_label_fraction: Minimum fraction of timesteps with active labels
            tau_p: Past window size (for encoder initialization)
            tau_f: Future window size (for prediction)
        """
        # Convert to integers to handle float inputs
        chunk_length = int(chunk_length)
        stride = int(stride) if stride is not None else max(1, chunk_length // 2)
        tau_p = int(tau_p)
        tau_f = int(tau_f)
        # Extract data for specified indices
        self.X = X.float()
        self.y = y.float()
        self.chunk_length = chunk_length
        self.stride = stride
        self.tau_p = tau_p
        self.tau_f = tau_f
        self.min_label_fraction = min_label_fraction
        print(f"📊 Creating ContinuousChunkDataset:")
        print(f"   • Data shape: X{self.X.shape}, y{self.y.shape}")
        print(f"   • Chunk length: {chunk_length}")
        print(f"   • Stride: {self.stride}")
        print(f"   • Min label fraction: {min_label_fraction}")
        # Find valid chunks
        self.valid_chunk_starts = self._find_valid_chunks()
        print(f"   • Valid chunks: {len(self.valid_chunk_starts)}")
        if len(self.valid_chunk_starts) > 0:
            coverage = len(self.valid_chunk_starts) * self.stride / len(self.X)
            print(f"   • Data coverage: {coverage:.1%}")
        self._split()
    
    def _find_valid_chunks(self):
        """Find chunks that meet the label activity criteria."""
        valid_starts = []
        max_start = len(self.X) - self.chunk_length - self.tau_f + 1
        if max_start <= 0:
            print(f"⚠️  Warning: Data too short for chunks of length {self.chunk_length}")
            return valid_starts
        for start in range(0, max_start, self.stride):
            y_chunk = self.y[start:start + self.chunk_length]
            active_timesteps = (y_chunk.sum(dim=1) > 0).float().mean() 
            if active_timesteps >= self.min_label_fraction:
                valid_starts.append(start)   
        return valid_starts
    
    def __len__(self):
        return len(self.valid_chunks)
    
    def __getitem__(self, idx):
        start_idx = self.valid_chunks[idx]
        end_idx = start_idx + self.chunk_length    
        # Extract chunk
        x_chunk = self.X[start_idx:end_idx]  # [chunk_len, n_neurons]
        y_chunk = self.y[start_idx:end_idx]  # [chunk_len, n_labels] 
        # Transpose to match expected format [n_neurons, chunk_len] and [n_labels, chunk_len]
        x_chunk = x_chunk.T  # [n_neurons, chunk_len]
        y_chunk = y_chunk.T  # [n_labels, chunk_len]
        return x_chunk, y_chunk
    
    def _split(self, test_split=0.2,  n_test_segments=4):
        total_length = self.X.shape[0]
        segment_starts = np.linspace(self.valid_chunk_starts[0], (self.valid_chunk_starts[-1] + self.chunk_length - self.valid_chunk_starts[0]), n_test_segments+1, dtype=int)
        train_indices = []
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        for i in range(len(segment_starts[:-1])):
            segment_length = segment_starts[i+1] - segment_starts[i]
            train_length = int(segment_length * (1 - test_split)) - (self.chunk_length + self.tau_p + self.tau_f)
            train_chunk_starts = np.arange(segment_starts[i], segment_starts[i] + train_length, self.stride, dtype=int)
            train_X.append(np.stack([self.X[start : start + self.chunk_length, :] for start in train_chunk_starts]))
            train_y.append(np.stack([self.y[start : start + self.chunk_length, :] for start in train_chunk_starts]))
            test_length = segment_length - train_length
            test_chunk_starts = np.arange(segment_starts[i] + train_length, segment_starts[i+1], self.stride, dtype=int)
            test_X.append(np.stack([self.X[start : start + self.chunk_length, :] for start in test_chunk_starts]))
            test_y.append(np.stack([self.y[start : start + self.chunk_length, :] for start in test_chunk_starts]))
        valid_train_chunks = (np.sum(np.sum(np.concatenate(train_y, axis=0), axis=1), axis=1) / self.chunk_length) > self.min_label_fraction
        self.train_X = np.concatenate(train_X, axis=0)[valid_train_chunks]
        self.train_y = np.concatenate(train_y, axis=0)[valid_train_chunks]
        self.test_X = np.concatenate(test_X, axis=0)
        self.test_y = np.concatenate(test_y, axis=0)
      
