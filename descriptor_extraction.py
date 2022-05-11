# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Authored by Gargi Balasubramaniam

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import pickle

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)

    ## FOR ISR Parsed Data
    parser.add_argument('--load_model_dir', type=str)
    parser.add_argument('--save_parsed_data_dir', type=str)

    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.

    in_splits = []
    out_splits = []

    # print(dataset)
    for env_i, env in enumerate(dataset):
        # print(env_i)#, env)

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights, uda_weights = None, None, None

        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)]


    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits )]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]


    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    algorithm.to(device)

    # IF group DRO, add this line
    algorithm.q = torch.zeros(3)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    evals = zip(eval_loader_names, eval_loaders, eval_weights)

    # Loading from path
    PATH = args.load_model_dir + "model.pkl"
    # PATH = "../ermnew/model.pkl"
    trained_model = algorithm
    trained_model.load_state_dict(torch.load(PATH)['model_dict'])

    feat_extr = trained_model.featurizer.to(device)    

    print(feat_extr)

    # Start Evaluation
    results = {}
    train_data = {}
    test_data = {} 

    train_x = []
    train_y = []
    train_z = []
    train_g = []
    train_preds = []
    train_logits = []

    test_x = []
    test_y = []
    test_z = []
    test_g = []
    test_preds = []
    test_logits = []
    torch.cuda.empty_cache()


    print("Starting to parse....")

    for name, loader, weights in evals:
        print("Current Split: ", name)
        acc = misc.accuracy(algorithm, loader, weights, device)

        x, y, z, p, l = misc.return_x_y_z(feat_extr, loader, weights, device, algorithm)
        g = torch.full(y.shape, int(name[3]))

        if 'in' in name:
            train_x.append(x)
            train_y.append(y)
            train_z.append(z)
            train_g.append(g)
            train_preds.append(p)
            train_logits.append(l)
        else:
            test_x.append(x)
            test_y.append(y)
            test_z.append(z)
            test_g.append(g)
            test_preds.append(p)
            test_logits.append(l)
        
    train_data['feature'] = torch.cat(train_z).detach().cpu().numpy()
    train_data['label'] = torch.cat(train_y).detach().cpu().numpy()
    train_data['pred'] = torch.cat(train_preds).detach().cpu().numpy()
    train_data['group'] = torch.cat(train_g).detach().cpu().numpy()
    train_data['logits'] = torch.cat(train_logits).detach().cpu().numpy()

    test_data['feature'] = torch.cat(test_z).detach().cpu().numpy()
    test_data['label'] = torch.cat(test_y).detach().cpu().numpy()
    test_data['pred'] = torch.cat(test_preds).detach().cpu().numpy()
    test_data['group'] = torch.cat(test_g).detach().cpu().numpy()
    test_data['logits'] = torch.cat(test_logits).detach().cpu().numpy()


    for key,val in test_data.items():
        print(key, val.shape)

    for key,val in train_data.items():
        print(key, val.shape)
        if key == 'group':
            print(val)

    save_dir = args.save_parsed_data_dir

    # Save Train Split
    fname =  f'groupdro_pacs_train_data.p'
    pickle.dump(train_data, open(save_dir + '/' + fname, 'wb'))

    # Save Test Split
    fname =  f'groupdro_pacs_test_data.p'
    pickle.dump(test_data, open(save_dir + '/' + fname, 'wb'))

       
    print("Finished! Parsed data has been saved to :", save_dir + '/' + fname)
