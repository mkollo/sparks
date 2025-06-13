import torch
from torch.utils.data import Dataset
import numpy as np

def create_stratified_temporal_split(X, y, n_test_segments=4, test_split=0.2, 
                                   chunk_length=1000, tau_f=1):
    """
    Create stratified temporal split with test segments distributed across recording
    to capture temporal non-stationarity while maintaining strict separation.
    
    Args:
        X: Neural data tensor [T, N_neurons]
        y: Target labels tensor [T, N_labels]
        n_test_segments: Number of test segments to distribute across recording
        test_split: Total fraction of data for testing
        chunk_length: Length of data chunks for training (how many timesteps per batch)
        tau_f: Future window size
        
    Returns:
        train_indices: List of indices for training data
        test_indices: List of indices for test data
    """
    # Convert to integers to handle float inputs
    chunk_length = int(chunk_length)
    tau_f = int(tau_f)
    
    total_length = len(X)
    segment_length = total_length // n_test_segments
    test_segment_size = int(segment_length * test_split)  # Each segment contributes equally
    buffer_size = chunk_length + tau_f
    
    test_indices = []
    train_indices = []
    
    print(f"🔄 Creating stratified temporal split:")
    print(f"   • Total length: {total_length:,}")
    print(f"   • Segments: {n_test_segments}")
    print(f"   • Test segment size: {test_segment_size:,}")
    print(f"   • Buffer size: {buffer_size}")
    
    for i in range(n_test_segments):
        # Define segment boundaries
        segment_start = i * segment_length
        segment_end = min((i + 1) * segment_length, total_length)
        
        # Test segment in middle of each segment
        test_start = segment_start + (segment_length - test_segment_size) // 2
        test_end = test_start + test_segment_size
        
        # Add buffer zones around test segment
        train_end_1 = test_start - buffer_size
        train_start_2 = test_end + buffer_size
        
        print(f"   • Segment {i+1}: [{segment_start:,}:{segment_end:,}]")
        print(f"     - Train: [{segment_start:,}:{train_end_1:,}] + [{train_start_2:,}:{segment_end:,}]")
        print(f"     - Test: [{test_start:,}:{test_end:,}]")
        
        # Collect indices (ensure boundaries are valid)
        if train_end_1 > segment_start:
            train_indices.extend(range(segment_start, max(segment_start, train_end_1)))
        if train_start_2 < segment_end:
            train_indices.extend(range(min(train_start_2, segment_end), segment_end))
            
        test_indices.extend(range(test_start, test_end))
    
    print(f"   • Final split: {len(train_indices):,} train, {len(test_indices):,} test")
    print(f"   • Test fraction: {len(test_indices)/total_length:.1%}")
    
    return train_indices, test_indices

class ContinuousChunkDataset(Dataset):
    """
    Dataset for continuous neural data that creates chunks from specified indices
    while filtering out chunks with insufficient label activity.
    """
    
    def __init__(self, X, y, indices, chunk_length, stride=None, 
                 filter_no_labels=True, min_label_fraction=0.1, tau_p=6, tau_f=1):
        """
        Args:
            X: Neural data tensor [T, N_neurons] 
            y: Target labels tensor [T, N_labels]
            indices: Specific timestep indices to use (from stratified split)
            chunk_length: Length of each data chunk for training
            stride: Step size between chunks (default: chunk_length // 2)
            filter_no_labels: Skip chunks with insufficient label activity
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
        self.X = X[indices].float()
        self.y = y[indices].float()
        self.chunk_length = chunk_length
        self.stride = stride
        self.tau_p = tau_p
        self.tau_f = tau_f
        self.filter_no_labels = filter_no_labels
        self.min_label_fraction = min_label_fraction
        
        print(f"📊 Creating ContinuousChunkDataset:")
        print(f"   • Data shape: X{self.X.shape}, y{self.y.shape}")
        print(f"   • Chunk length: {chunk_length}")
        print(f"   • Stride: {self.stride}")
        print(f"   • Filter no-labels: {filter_no_labels}")
        print(f"   • Min label fraction: {min_label_fraction}")
        
        # Find valid chunks
        self.valid_chunks = self._find_valid_chunks()
        
        print(f"   • Valid chunks: {len(self.valid_chunks)}")
        if len(self.valid_chunks) > 0:
            coverage = len(self.valid_chunks) * self.stride / len(self.X)
            print(f"   • Data coverage: {coverage:.1%}")
    
    def _find_valid_chunks(self):
        """Find chunks that meet the label activity criteria."""
        valid_starts = []
        max_start = len(self.X) - self.chunk_length - self.tau_f + 1
        
        if max_start <= 0:
            print(f"⚠️  Warning: Data too short for chunks of length {self.chunk_length}")
            return valid_starts
        
        for start in range(0, max_start, self.stride):
            if self.filter_no_labels:
                # Check label activity in this chunk
                y_chunk = self.y[start:start + self.chunk_length]
                active_timesteps = (y_chunk.sum(dim=1) > 0).float().mean()
                
                if active_timesteps >= self.min_label_fraction:
                    valid_starts.append(start)
            else:
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

def create_continuous_datasets(X, y, chunk_length=1000, n_test_segments=4, 
                             test_split=0.2, stride=None, min_label_fraction=0.1,
                             tau_p=6, tau_f=1):
    """
    Create train/test datasets from continuous neural data with stratified temporal splitting.
    
    Args:
        X: Neural data [T, N_neurons]
        y: Labels [T, N_labels]
        chunk_length: Length of data chunks for training
        n_test_segments: Number of test segments distributed across recording
        test_split: Total fraction for test set
        stride: Step between chunks (default: chunk_length // 2)
        min_label_fraction: Minimum fraction of timesteps with labels per chunk
        tau_p: Past window size
        tau_f: Future window size
    
    Returns:
        train_dataset, test_dataset
    """
    print(f"🔄 Creating continuous datasets from data:")
    print(f"   • X shape: {X.shape}")
    print(f"   • y shape: {y.shape}")
    print(f"   • Chunk length: {chunk_length}")
    
    # Create stratified temporal split
    train_indices, test_indices = create_stratified_temporal_split(
        X, y, n_test_segments, test_split, chunk_length=chunk_length, tau_f=tau_f
    )
    
    # Create datasets
    train_dataset = ContinuousChunkDataset(
        X, y, train_indices, chunk_length, stride, 
        filter_no_labels=True, min_label_fraction=min_label_fraction,
        tau_p=tau_p, tau_f=tau_f
    )
    
    test_dataset = ContinuousChunkDataset(
        X, y, test_indices, chunk_length, stride,
        filter_no_labels=True, min_label_fraction=min_label_fraction, 
        tau_p=tau_p, tau_f=tau_f
    )
    
    print(f"✅ Datasets created successfully!")
    
    return train_dataset, test_dataset

def compute_masked_loss(decoder_outputs, targets, loss_fn):
    """
    Compute loss only for timesteps with active labels.
    
    Args:
        decoder_outputs: Model predictions [batch_size, n_labels]
        targets: Target labels [batch_size, n_labels] 
        loss_fn: Loss function
        
    Returns:
        loss: Computed loss (0 if no active timesteps)
    """
    # Identify timesteps with any active label
    active_mask = (targets.sum(dim=-1) > 0)  # [batch_size]
    
    if not active_mask.any():
        return torch.tensor(0.0, device=targets.device, requires_grad=True)
    
    # Apply mask
    active_outputs = decoder_outputs[active_mask]
    active_targets = targets[active_mask]
    
    return loss_fn(active_outputs, active_targets)
