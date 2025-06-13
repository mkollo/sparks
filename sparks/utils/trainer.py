import os
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.auto import trange

from sparks.utils.vae import skip, ae_forward
from sparks.utils.continuous_dataset import compute_masked_loss

class SparksTrainer:
    def __init__(self, 
                 encoder, 
                 decoder, 
                 train_data, 
                 test_data=None,
                 latent_dim=None,
                 tau_p=None, 
                 tau_f=None, 
                 loss_fn=None, 
                 optimizer=None,
                 out_folder="models", 
                 batch_size=12, 
                 device=None, 
                 local_rank=None,
                 test_split=0.2,
                 online=False,
                 use_masked_loss=False):
        """
            encoder: The encoder model
            decoder: The decoder model
            train_data: Either a Dataset object or a tuple of (X, y) tensors
            test_data: Either a Dataset object or a tuple of (X, y) tensors. If None and train_data is a tuple,
                       a portion of train_data will be used for testing based on test_split
            latent_dim: Dimension of the latent space
            tau_p: Size of the past window
            tau_f: Size of the future window
            loss_fn: Loss function
            optimizer: Optimizer
            out_folder: Folder to save models and logs
            batch_size: Batch size for training
            device: Device to use (if None, will be automatically detected)
            local_rank: Local rank for distributed training
            test_split: Fraction of data to use for testing if test_data is None and train_data is a tuple
            online: If True, performs gradient updates at every timestep instead of accumulating across sequence.
                   This is more memory efficient and biologically plausible but may require learning rate adjustment.
            use_masked_loss: If True, skips timesteps where no labels are active (all targets are 0).
                           Useful for continuous data training to prevent bias toward inactive states.
        """
        # Setup device and distributed training
        self.setup_environment(device, local_rank)
        
        # Setup models
        self.setup_models(encoder, decoder)
        
        # Setup hyperparameters
        self.latent_dim = latent_dim
        self.tau_p = tau_p
        self.tau_f = tau_f
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.online = online
        self.use_masked_loss = use_masked_loss

        # Setup data loaders
        self.setup_data_loaders(train_data, test_data, batch_size, test_split)

        # Setup output directories and logging
        self.setup_output_directories(out_folder)
        self.training_log = []
        self.fig_frames = []
        
        # Warn about online mode considerations
        if self.online and self._is_main():
            print("⚠️  Online mode enabled: Consider reducing learning rate as gradients are updated at every timestep.")

    def setup_environment(self, device, local_rank):
        """Setup the computing environment (CPU, MPS, CUDA, multi-GPU)."""
        # Determine local rank and distributed setup
        if local_rank is not None:
            self.local_rank = int(local_rank)
        else:
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))        
        
        # Initialize distributed training if needed
        if self.local_rank >= 0 and not dist.is_initialized() and local_rank is not None and torch.cuda.is_available():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            
        self.use_ddp = (self.local_rank >= 0 and self._is_distributed())
        
        # Determine device
        if device is None:
            if self.use_ddp:
                self.device = torch.device(f"cuda:{self.local_rank}")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    def setup_models(self, encoder, decoder):
        """Setup the encoder and decoder models."""
        if self.use_ddp:
            self.encoder = DDP(encoder.to(self.device), device_ids=[self.local_rank], find_unused_parameters=False)
            self.decoder = DDP(decoder.to(self.device), device_ids=[self.local_rank], find_unused_parameters=False)
        else:
            self.encoder = encoder.to(self.device)
            self.decoder = decoder.to(self.device)
            
        # Ensure reparametrize is accessible when using DDP
        if isinstance(self.encoder, torch.nn.parallel.DistributedDataParallel):
            self.encoder.reparametrize = self.encoder.module.reparametrize

    def _should_use_device_transfer(self):
        """Determine if we should transfer data to accelerated device (GPU/MPS)"""
        return self.device.type != 'cpu'

    def _create_device_collate_fn(self):
        """Create collate function that automatically moves data to device (GPU/MPS)"""
        def device_collate_fn(batch):
            inputs, targets = torch.utils.data.dataloader.default_collate(batch)
            return (
                inputs.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True)
            )
        return device_collate_fn

    def setup_data_loaders(self, train_data, test_data, batch_size, test_split):
        """
        Setup data loaders for training and testing.
        Supports both Dataset objects and (X, y) tensor tuples.
        """
        # Handle X, y tensor inputs
        if isinstance(train_data, tuple) and len(train_data) == 2:
            X_train, y_train = train_data
            
            # Create dataset from tensors - KEEP ON CPU
            train_dataset = TensorDataset(
                X_train.float(),  # Keep on CPU
                y_train.float()   # Keep on CPU
            )
            
            # Split into train/test if test_data not provided
            if test_data is None:
                train_size = int((1 - test_split) * len(train_dataset))
                test_size = len(train_dataset) - train_size
                train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
            else:
                X_test, y_test = test_data
                test_dataset = TensorDataset(
                    X_test.float(),   # Keep on CPU
                    y_test.float()    # Keep on CPU
                )
        else:
            # Use provided datasets
            train_dataset = train_data
            test_dataset = test_data if test_data is not None else train_data
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        # Setup samplers for distributed training
        if self.use_ddp:
            self.train_sampler = DistributedSampler(train_dataset)
            self.test_sampler = DistributedSampler(test_dataset)
        else:
            self.train_sampler = None
            self.test_sampler = None
        
        # Setup device-aware data loading
        use_device_transfer = self._should_use_device_transfer()
        collate_fn = self._create_device_collate_fn() if use_device_transfer else None
            
        # Create data loaders with automatic device transfer
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=not self.train_sampler,
            pin_memory=use_device_transfer,  # Only pin memory if transferring to accelerated device
            num_workers=min(4, os.cpu_count()-2),
            collate_fn=collate_fn  # Automatic device transfer
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            sampler=self.test_sampler,
            shuffle=not self.test_sampler,
            pin_memory=use_device_transfer,
            num_workers=min(4, os.cpu_count()-2),
            collate_fn=collate_fn  # Automatic device transfer
        )

    def setup_output_directories(self, out_folder):
        """
        Setup output directories for models, logs, and visualizations.        
        """
        self.out_folder = out_folder
        self.embedding_dir = os.path.join(out_folder, "embeddings")
        
        # Only create directories on the main process
        if self._is_main():
            os.makedirs(self.out_folder, exist_ok=True)
            os.makedirs(self.embedding_dir, exist_ok=True)
        
        # Add synchronization barrier to ensure directories are created
        if self.use_ddp:
            dist.barrier()
            
        self.log_path = os.path.join(out_folder, "training_log.csv")
        self.gif_path = os.path.join(out_folder, "training_progress.gif")

    def _is_distributed(self):
        """Check if distributed training is available and initialized."""
        return dist.is_available() and dist.is_initialized()

    def _is_main(self):
        """Check if this is the main process."""
        return not self.use_ddp or dist.get_rank() == 0

    def save_models(self, epoch):
        """Save model checkpoints."""
        if self._is_main():
            # Save model
            if isinstance(self.encoder, DDP):
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.module.state_dict() if isinstance(self.encoder, DDP) else self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.module.state_dict() if isinstance(self.decoder, DDP) else self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.out_folder, f"checkpoint_epoch_{epoch}.pth"))
            else:
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict() if isinstance(self.encoder, DDP) else self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict() if isinstance(self.decoder, DDP) else self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.out_folder, f"checkpoint_epoch_{epoch}.pth"))       
            
    @torch.no_grad()
    def test_on_batch(self, inputs, targets, sess_id=0):
        """
        Test the model on a single batch.
        Note: inputs and targets are already on the correct device via DataLoader collate_fn
        """
        self.encoder.eval()
        self.decoder.eval()
        
        encoder_outputs = torch.zeros([len(inputs), self.latent_dim, self.tau_p], device=self.device)
        test_loss = 0
        all_decoder_outputs = []

        inputs, targets = skip(self.encoder, inputs, targets, self.device, num_steps=0, sess_id=sess_id)
        T = inputs.shape[-1]

        for t in range(T):
            encoder_outputs, decoder_outputs, _, _ = ae_forward(
                self.encoder, self.decoder, inputs[..., t], encoder_outputs, self.tau_p, self.device, sess_id
            )
            all_decoder_outputs.append(decoder_outputs.unsqueeze(-1))

            if self.loss_fn is not None and t < T - self.tau_f + 1:
                target = targets[..., t:t + self.tau_f].reshape(targets.shape[0], -1)
                test_loss += self.loss_fn(decoder_outputs, target).cpu() / T

        decoder_outputs = torch.cat(all_decoder_outputs, dim=-1)
        return test_loss, encoder_outputs.cpu(), decoder_outputs.cpu()

    def evaluate(self, dependent_keys=None, sess_id=0):
        """
        Evaluate the model on the test dataset.        
        """
        all_encoder_outputs, all_decoder_outputs = [], []
        total_loss = 0
        all_targets = []

        for inputs, targets in self.test_loader:
            loss, enc_out, dec_out = self.test_on_batch(inputs, targets, sess_id=sess_id)
            total_loss += loss
            all_encoder_outputs.append(enc_out)
            all_decoder_outputs.append(dec_out)
            all_targets.append(targets.cpu())

        # Concatenate outputs
        encoder_outputs = torch.cat(all_encoder_outputs)
        decoder_outputs = torch.cat(all_decoder_outputs)
        y_true = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate accuracy
        probs = torch.sigmoid(decoder_outputs).numpy()
        y_pred_bin = (probs > 0.5).astype(np.float32)
        y_true_bin = y_true.astype(np.float32)

        per_feature_acc = []
        # Calculate per-feature accuracy
        for f in range(y_true.shape[1]):
            try:
                acc = accuracy_score(
                    y_true_bin[:, f, :].reshape(-1),
                    y_pred_bin[:, f, :].reshape(-1)
                )
                per_feature_acc.append(acc)
            except:
                # Handle case where y_true or y_pred is empty
                per_feature_acc.append(0.0)
        
        return total_loss.item(), per_feature_acc, y_true.shape[1]

    def log_and_plot(self, epoch, test_loss, per_feature_acc, dependent_keys):
        """
        Log training progress and create visualizations.        
        """
        if not self._is_main():
            return

        # If dependent_keys not provided, create generic keys
        if dependent_keys is None:
            dependent_keys = [f"feature_{i}" for i in range(len(per_feature_acc))]

        # Log metrics
        mean_acc = np.mean(per_feature_acc)
        self.training_log.append({
            "epoch": epoch + 1,
            "test_loss": test_loss,
            "mean_accuracy": mean_acc,
            **{f"acc_{dependent_keys[i]}": per_feature_acc[i] for i in range(len(dependent_keys))}
        })

        # Create visualizations every 5 epochs
        if (epoch + 1) % 5 == 0:
            self.save_models(epoch + 1)
            
            # Create plots
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            df = pd.DataFrame(self.training_log)
            
            # Plot test loss
            ax[0].plot(df["epoch"], df["test_loss"], label="Test Loss")
            ax[0].set_title("Test Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            
            # Plot per-feature accuracy
            ax[1].bar(dependent_keys, per_feature_acc)
            ax[1].set_title(f"Per-feature accuracy (Epoch {epoch+1})")
            ax[1].set_xticklabels(dependent_keys, rotation=90)
            ax[1].set_ylim(0, 1.0)
            
            plt.tight_layout()
            
            # Save figure as frame for GIF
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.fig_frames.append(Image.fromarray(img))
            
            # Save log and GIF
            df.to_csv(self.log_path, index=False)
            if self.fig_frames:
                self.fig_frames[0].save(
                    self.gif_path,
                    save_all=True,
                    append_images=self.fig_frames[1:],
                    duration=300,
                    loop=0
                )
            plt.close(fig)

    def fit(self, n_epochs, beta=0.001, dependent_keys=None, sess_id=0):
        """
        Train the model.
                
            n_epochs: Number of epochs
            beta: Regularization parameter (kept for compatibility but not used in loss)
            dependent_keys: List of keys for dependent variables
            sess_id: Session ID
        """
        encoder = self.encoder
        decoder = self.decoder
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device
        tau_p = self.tau_p
        tau_f = self.tau_f
        train_loader = self.train_loader

        try:
            for epoch in trange(n_epochs, desc="Training Epochs", disable=not self._is_main()):
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                encoder.train()
                decoder.train()
                total_loss = 0
                num_sequences = 0
                for inputs, targets in train_loader:
                    # Note: inputs and targets are already on the correct device via DataLoader collate_fn
                    inputs, targets = skip(encoder, inputs, targets, device, sess_id=sess_id)
                    encoder_outputs = torch.zeros(inputs.size(0), self.latent_dim, tau_p, device=device)

                    if not self.online:
                        optimizer.zero_grad()
                    
                    batch_loss = 0
                    for t in range(inputs.shape[-1]):
                        encoder_outputs, decoder_outputs, _, _ = ae_forward(
                            encoder, decoder, inputs[..., t], encoder_outputs, tau_p, device, sess_id
                        )
                        if t < inputs.shape[-1] - tau_f + 1:
                            target = targets[..., t:t + tau_f].reshape(targets.shape[0], -1)
                            
                            # Use masked loss if enabled (skip timesteps with no active labels)
                            if self.use_masked_loss:
                                loss = compute_masked_loss(decoder_outputs, target, loss_fn)
                            else:
                                loss = loss_fn(decoder_outputs, target)
                            
                            if self.online:
                                # Online mode: update weights at every timestep
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                # Detach from computational graph to save memory
                                encoder.detach_()
                                encoder_outputs = encoder_outputs.detach()
                            else:
                                # Standard mode: accumulate gradients
                                loss.backward()
                            
                            batch_loss += loss.item()
                    
                    total_loss += batch_loss
                    num_sequences += inputs.size(0)
                    
                    if not self.online:
                        optimizer.step()
                avg_loss = total_loss / num_sequences
                # Synchronize before evaluation
                if self.use_ddp:
                    dist.barrier()

                # Evaluate and log
                test_loss, per_feature_acc, _ = self.evaluate(dependent_keys, sess_id)
                self.log_and_plot(epoch, avg_loss, per_feature_acc, dependent_keys)

            if self._is_main():
                print(f"\n✅ Training complete. Log saved to {self.log_path}, GIF saved to {self.gif_path}")
                
        finally:
            # Clean up distributed process group
            if self.use_ddp and dist.is_initialized():
                dist.destroy_process_group()
