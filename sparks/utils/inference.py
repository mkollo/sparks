import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SparksInference:
    def __init__(self, encoder, decoder, tau_p, device=None, batch_size=12):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.encoder = encoder.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.tau_p = tau_p
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(cls, checkpoint_path, encoder, decoder, tau_p, device=None, batch_size=12):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        return cls(encoder, decoder, tau_p, device, batch_size)

    @torch.no_grad()
    def infer_windows(self, spike_data, stride=1):
        """
        spike_data: Tensor of shape [N, L], where N = n_neurons, L = total bins
        stride: how many bins to move for each new window
        Returns:
            embeddings: Tensor of shape [num_windows, latent_dim]
        """
        N, L = spike_data.shape
        T = self.tau_p
        # Number of windows
        num_windows = max(0, (L - T) // stride + 1)
        window_list = []

        # Construct all windows
        for i in range(0, L - T + 1, stride):
            window = spike_data[:, i : i + T]       # [N, T]
            window_list.append(window)
        # Stack and create DataLoader
        all_windows = torch.stack(window_list, dim=0)  # [num_windows, N, T]
        loader = DataLoader(all_windows, batch_size=self.batch_size, shuffle=False)

        latent_list = []
        for batch_windows in loader:
            # Move to device and permute if necessary to match encoder's input shape
            # If encoder expects [batch, n_neurons, T] exactly, then:
            batch_windows = batch_windows.to(self.device).float()
            # Run encoder
            mu, logvar = self.encoder(batch_windows, sess_id=0)
            z = self.encoder.reparametrize(mu, logvar)   # [batch_size, latent_dim]
            latent_list.append(z.cpu())
        embeddings = torch.cat(latent_list, dim=0)  # [num_windows, latent_dim]
        return embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        """
        latents: Tensor [num_points, latent_dim]
        returns: decoded outputs. If the decoder reconstructs [N, T], output will be [num_points, N, T].
        """
        outputs = []
        for i in range(0, latents.shape[0], self.batch_size):
            chunk = latents[i : i + self.batch_size].to(self.device)
            recon = self.decoder(chunk)  # shape depends on decoder definition
            outputs.append(recon.cpu())
        return torch.cat(outputs, dim=0)
