import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm.auto import trange

from sparks.utils.train import train
from sparks.utils.misc import identity
from sparks.utils.vae import skip, ae_forward

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank

def make_dataloaders(X, y, batch_size, device):
    dataset = TensorDataset(X.float().to(device), y.float().to(device))
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader, train_sampler, test_sampler

@torch.no_grad()
def test_on_batch(encoder, decoder, inputs, targets, latent_dim, tau_p, tau_f=1,
                  loss_fn=None, device='cpu', burnin=0, sess_id=0):
    encoder.eval()
    encoder_outputs = torch.zeros([len(inputs), latent_dim, tau_p], device=device)
    test_loss = 0
    all_decoder_outputs = []

    inputs, targets = skip(encoder, inputs, targets, device, num_steps=burnin, sess_id=sess_id)
    T = inputs.shape[-1]

    for t in range(T):
        encoder_outputs, decoder_outputs, _, _ = ae_forward(
            encoder, decoder, inputs[..., t], encoder_outputs, tau_p, device, sess_id
        )
        all_decoder_outputs.append(decoder_outputs.unsqueeze(-1))

        if loss_fn is not None and t < T - tau_f + 1:
            target = targets[..., t:t + tau_f].reshape(targets.shape[0], -1).to(device)
            test_loss += loss_fn(decoder_outputs, target).cpu() / T

    decoder_outputs = torch.cat(all_decoder_outputs, dim=-1)
    return test_loss, encoder_outputs.cpu(), decoder_outputs.cpu()

def test(encoder, decoder, test_dls, latent_dim, tau_p, tau_f=1,
         loss_fn=None, device='cpu', sess_ids=None, **kwargs):
    all_encoder_outputs, all_decoder_outputs = [], []
    total_loss = 0
    if sess_ids is None:
        sess_ids = np.arange(len(test_dls))

    for i, test_dl in enumerate(test_dls):
        for inputs, targets in test_dl:
            loss, enc_out, dec_out = test_on_batch(
                encoder, decoder, inputs, targets, latent_dim, tau_p,
                tau_f=tau_f, loss_fn=loss_fn, device=device, sess_id=sess_ids[i], **kwargs
            )
            total_loss += loss
            all_encoder_outputs.append(enc_out)
            all_decoder_outputs.append(dec_out)

    return total_loss, torch.cat(all_encoder_outputs), torch.cat(all_decoder_outputs)