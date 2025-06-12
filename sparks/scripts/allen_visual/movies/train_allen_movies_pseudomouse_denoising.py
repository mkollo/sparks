import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.movies_pseudomouse import make_pseudomouse_allen_movies_dataset
from sparks.scripts.allen_visual.movies.utils.test import test
from sparks.utils.train import train_on_batch
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.misc import make_res_folder, save_results

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=9, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=256, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp',
                        help='Type of decoder (one of linear, mlp or deconv)')
    parser.add_argument('--output_type', type=str, default='flatten',
                          help='Output architecture for the decoder')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=0.5, help='STDP decay')
    parser.add_argument('--w_pre', type=float, default=0.1, help='')
    parser.add_argument('--w_post', type=float, default=0.05, help='')

    # Data parameters
    parser.add_argument('--data_type', type=str, default='ephys', choices=['ephys', 'calcium'],
                        help='Whether to use neuropixels or calcium data')
    parser.add_argument('--n_neurons', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')

    # sliding
    parser.add_argument('--block_size', type=int, default=100, help='Dimension of the sliding attention blocks')
    parser.add_argument('--window_size', type=int, default=3, help='Size of the sliding window')
    parser.add_argument('--sliding', action='store_true', default=False, help='')


    args = parser.parse_args()

    # Create folder to save results
    make_res_folder('allen_movies_pseudomouse_denoising' + '_nneurons_' + str(args.n_neurons),
                    os.getcwd(), args)

    (train_dataset_1, test_dataset_1,
    train_dl_1, test_dl_1) = make_pseudomouse_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                                   n_neurons=args.n_neurons,
                                                                   dt=args.dt,
                                                                   block='first',
                                                                   batch_size=args.batch_size,
                                                                   num_workers=args.num_workers,
                                                                   mode='unsupervised',
                                                                   ds=args.ds,
                                                                   seed=args.seed)

    (train_dataset_2, test_dataset_2,
     train_dl_2, test_dl_2) = make_pseudomouse_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                                    n_neurons=args.n_neurons,
                                                                    dt=args.dt,
                                                                    block='second',
                                                                    batch_size=args.batch_size,
                                                                    num_workers=args.num_workers,
                                                                    mode='unsupervised',
                                                                    ds=args.ds,
                                                                    correct_units_ids=train_dataset_1.good_units_ids,
                                                                    seed=args.seed)

    np.save(args.results_path + '/good_units_ids.npy', train_dataset_1.good_units_ids)

    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=args.n_neurons,
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads,
                                                 output_type=args.output_type,
                                                 sliding=args.sliding,
                                                 window_size=args.window_size,
                                                 block_size=args.block_size,
                                                 w_pre=args.w_pre,
                                                 w_post=args.w_post).to(args.device)

    decoding_network = get_decoder(output_dim_per_session=args.n_neurons * args.tau_f, args=args, softmax=False)

    if args.online:
        args.lr = args.lr / 900
    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = -np.inf

    for epoch in tqdm.tqdm(range(args.n_epochs)):
        train_iter_1 = iter(train_dl_1)
        train_iter_2 = iter(train_dl_2)
        for (inputs_1, targets_1), (inputs_2, targets_2) in zip(train_iter_1, train_iter_2):
            inputs = torch.cat((inputs_1, inputs_2), dim=0)
            targets = torch.cat((targets_2, targets_1), dim=0)
            train_on_batch(encoder=encoding_network, decoder=decoding_network,
                           inputs=inputs, targets=targets,
                           loss_fn=loss_fn, optimizer=optimizer,
                           latent_dim=args.latent_dim,
                           tau_p=args.tau_p, tau_f=args.tau_f,
                           device=args.device, online=args.online,
                           beta=args.beta)

        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_acc, encoder_outputs, decoder_outputs = test(encoder=encoding_network,
                                                              decoder=decoding_network,
                                                              test_dls=[test_dl_1, test_dl_2],
                                                              true_frames=test_dataset_1.true_frames,
                                                              mode='unsupervised',
                                                              latent_dim=args.latent_dim,
                                                              tau_p=args.tau_p,
                                                              tau_f=args.tau_f,
                                                              loss_fn=loss_fn,
                                                              sess_ids=[0, 0],
                                                              device=args.device)
            best_test_acc = save_results(args.results_path, test_acc, best_test_acc, encoder_outputs,
                                         decoder_outputs, encoding_network, decoding_network)

            print("Epoch %d, test loss: %.3f" % (epoch, -test_acc))
