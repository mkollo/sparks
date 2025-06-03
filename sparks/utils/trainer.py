import os
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.auto import trange

from sparks.utils.vae import skip, ae_forward
from sparks.utils.misc import identity


class SparksTrainer:
    def __init__(self, encoder, decoder, train_dataset, test_dataset,
                 latent_dim, tau_p, tau_f, loss_fn, optimizer,
                 out_folder="models", batch_size=12, device=None, local_rank=None):

        self.device = (
            torch.device('cuda') if torch.cuda.is_available() else
            torch.device('mps')  if torch.backends.mps.is_available() else
            torch.device('cpu')
        )
        self.local_rank = local_rank or int(os.environ.get("LOCAL_RANK", -1))
        self.use_ddp = (local_rank is not None and local_rank >= 0 and self._is_distributed())
        
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            self.encoder = DDP(encoder.to(self.device), device_ids=[self.local_rank], find_unused_parameters=True)
            self.decoder = DDP(decoder.to(self.device), device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            self.encoder = encoder.to(self.device)
            self.decoder = decoder.to(self.device)
            
        if isinstance(encoder, torch.nn.parallel.DistributedDataParallel):
            encoder.reparametrize = encoder.module.reparametrize
            
        self.latent_dim = latent_dim
        self.tau_p = tau_p
        self.tau_f = tau_f
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if self.use_ddp:
            self.train_sampler = DistributedSampler(train_dataset)
            self.test_sampler = DistributedSampler(test_dataset)
        else:
            self.train_sampler = None
            self.test_sampler = None
            
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               sampler=self.train_sampler if self.train_sampler else None,
                               shuffle=not self.train_sampler)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    sampler=self.test_sampler if self.test_sampler else None,
                                    shuffle=not self.test_sampler)

        self.out_folder = out_folder
        self.embedding_dir = os.path.join(out_folder, "embeddings")
        os.makedirs(self.embedding_dir, exist_ok=True)
        self.log_path = os.path.join(out_folder, "training_log.csv")
        self.gif_path = os.path.join(out_folder, "training_progress.gif")
        self.training_log = []
        self.fig_frames = []

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized()

    def save_models(self, epoch):
        if not self.use_ddp or dist.get_rank() == 0:
            torch.save(self.encoder.state_dict(), os.path.join(self.out_folder, f"encoder_epoch_{epoch}.pth"))
            torch.save(self.decoder.state_dict(), os.path.join(self.out_folder, f"decoder_epoch_{epoch}.pth"))

    def evaluate(self, dependent_keys):
        from sparks.utils.test import test
        test_loss, encoder_outputs, decoder_outputs = test(
            encoder=self.encoder,
            decoder=self.decoder,
            test_dls=[self.test_loader],
            loss_fn=self.loss_fn,
            latent_dim=self.latent_dim,
            tau_p=self.tau_p,
            tau_f=self.tau_f,
            device=self.device,
        )

        y_true = torch.cat([y for _, y in self.test_loader], dim=0).cpu().numpy()
        probs = torch.sigmoid(decoder_outputs).cpu().numpy()
        y_pred_bin = (probs > 0.5).astype(np.float32)
        y_true_bin = y_true.astype(np.float32)

        per_feature_acc = [accuracy_score(
            y_true_bin[:, f, :].reshape(-1),
            y_pred_bin[:, f, :].reshape(-1))
            for f in range(y_true.shape[1])
        ]

        return test_loss.item(), per_feature_acc, y_true.shape[1]

    def log_and_plot(self, epoch, test_loss, per_feature_acc, dependent_keys):
        is_main = not self.use_ddp or dist.get_rank() == 0
        if not is_main:
            return

        mean_acc = np.mean(per_feature_acc)
        self.training_log.append({
            "epoch": epoch + 1,
            "test_loss": test_loss,
            "mean_accuracy": mean_acc,
            **{f"acc_{dependent_keys[i]}": per_feature_acc[i] for i in range(len(dependent_keys))}
        })

        if (epoch + 1) % 5 == 0:
            self.save_models(epoch + 1)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            df = pd.DataFrame(self.training_log)
            ax[0].plot(df["epoch"], df["test_loss"], label="Test Loss")
            ax[0].set_title("Test Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[1].bar(dependent_keys, per_feature_acc)
            ax[1].set_title(f"Per-feature accuracy (Epoch {epoch+1})")
            ax[1].set_xticklabels(dependent_keys, rotation=90)
            ax[1].set_ylim(0, 1.0)
            plt.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.fig_frames.append(Image.fromarray(img))
            df.to_csv(self.log_path, index=False)
            self.fig_frames[0].save(
                self.gif_path,
                save_all=True,
                append_images=self.fig_frames[1:],
                duration=300,
                loop=0
            )
            plt.close(fig)

    def fit(self, n_epochs, beta=0.001, dependent_keys=None):
        encoder = self.encoder
        decoder = self.decoder
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device
        tau_p = self.tau_p
        tau_f = self.tau_f
        train_loader = self.train_loader

        for epoch in trange(n_epochs, desc="Training Epochs", disable=not self._is_main()):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            encoder.train()
            decoder.train()
            total_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = skip(encoder, inputs, targets, device)
                encoder_outputs = torch.zeros(inputs.size(0), self.latent_dim, tau_p, device=device)

                optimizer.zero_grad()
                for t in range(inputs.shape[-1]):
                    encoder_outputs, decoder_outputs, _, _ = ae_forward(
                        encoder, decoder, inputs[..., t], encoder_outputs, tau_p, device
                    )
                    if t < inputs.shape[-1] - tau_f + 1:
                        target = targets[..., t:t + tau_f].reshape(targets.shape[0], -1)
                        loss = loss_fn(decoder_outputs, target)
                        loss.backward()
                        total_loss += loss.item()
                optimizer.step()

            test_loss, per_feature_acc, _ = self.evaluate(dependent_keys)
            self.log_and_plot(epoch, test_loss, per_feature_acc, dependent_keys)

        if self._is_main():
            print(f"\n✅ Training complete. Log saved to {self.log_path}, GIF saved to {self.gif_path}")
        if self.use_ddp:
            dist.destroy_process_group()

    def _is_main(self):
        return not self.use_ddp or dist.get_rank() == 0