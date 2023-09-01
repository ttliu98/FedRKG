# -*- coding: utf-8 -*-
"""define all global parameters here."""
import argparse
import os
import random
import socket
import time
import traceback
from contextlib import closing
from os.path import join
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.distributed as dist
import yaml

import pcode.models as models
from pcode.utils import topology, param_parser, checkpoint, logging
from pcode.utils.param_parser import str2bool

def get_args():
    parser = add_argument()
    conf=parse_args_with_yaml(parser)
    set_environ()
    debug_parameter(conf)
    complete_missing_config(conf)
    experiment_paramenter(conf)
    validate_config(conf)
    return conf


def add_argument():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")
    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )
    # feed them to the parser.
    parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")
    # add arguments.
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)
    # dataset.
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument("--val_data_ratio", type=float, default=0)
    parser.add_argument(
        "--train_data_ratio", type=float, default=0, help="after the train/val split."
    )
    parser.add_argument(
        "--data_dir", default=RAW_DATA_DIRECTORY, help="path to dataset"
    )
    parser.add_argument("--img_resolution", type=int, default=None)
    parser.add_argument("--use_fake_centering", type=str2bool, default=False)
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument(
        "--partition_data",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument("--pin_memory", default=True, type=str2bool)
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--pn_normalize", default=True, type=str2bool, help="normalize by mean/std."
    )
    # model
    parser.add_argument(
        "--arch",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    parser.add_argument(
        "--complex_arch", type=str, default="master=resnet20,worker=resnet8:resnet14",
        help="specify the model when master and worker are not the same",
    )
    parser.add_argument("--w_conv_bias", default=False, type=str2bool)
    parser.add_argument("--w_fc_bias", default=True, type=str2bool)
    parser.add_argument("--freeze_bn", default=False, type=str2bool)
    parser.add_argument("--freeze_bn_affine", default=False, type=str2bool)
    parser.add_argument("--resnet_scaling", default=1, type=float)
    parser.add_argument("--vgg_scaling", default=None, type=int)
    parser.add_argument("--evonorm_version", default=None, type=str)
    # data, training and learning scheme.
    parser.add_argument("--n_comm_rounds", type=int, default=90)
    parser.add_argument(
        "--target_perf", type=float, default=None, help="it is between [0, 100]."
    )
    parser.add_argument("--early_stopping_rounds", type=int, default=0)
    parser.add_argument("--local_n_epochs", type=int, default=1)
    parser.add_argument("--random_reinit_local_model", default=None, type=str)
    parser.add_argument("--local_prox_term", type=float, default=0)
    parser.add_argument("--min_local_epochs", type=float, default=None)
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument(
        "--batch_size",
        "-b",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument("--base_batch_size", default=None, type=int)
    parser.add_argument(
        "--n_clients",
        default=1,
        type=int,
        help="# of the clients for federated learning.",
    )
    parser.add_argument(
        "--participation_ratio",
        default=0.1,
        type=float,
        help="number of participated ratio per communication rounds",
    )
    parser.add_argument("--n_participated", default=None, type=int)
    parser.add_argument("--fl_aggregate", default=None, type=str)
    parser.add_argument("--non_iid_alpha", default=0, type=float)
    parser.add_argument("--train_fast", type=str2bool, default=False)
    parser.add_argument("--use_mixup", default=False, type=str2bool)
    parser.add_argument("--mixup_alpha", default=1.0, type=float)
    parser.add_argument("--mixup_noniid", default=False, type=str2bool)
    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="MultiStepLR",
        choices=["MultiStepLR", "ExponentialLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--lr_milestones", type=str, default=None)
    parser.add_argument("--lr_milestone_ratios", type=str, default=None)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_init_lr", type=float, default=None)
    parser.add_argument("--lr_scaleup_factor", type=int, default=None)
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_epochs_upper_bound", type=int, default=150)
    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd")
    # quantizer
    parser.add_argument("--local_model_compression", type=str, default=None)
    # some SOTA training schemes, e.g., larc, label smoothing.
    parser.add_argument("--use_larc", type=str2bool, default=False)
    parser.add_argument("--larc_trust_coefficient", default=0.02, type=float)
    parser.add_argument("--larc_clip", default=True, type=str2bool)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--weighted_loss", default=None, type=str)
    parser.add_argument("--weighted_beta", default=0, type=float)
    parser.add_argument("--weighted_gamma", default=0, type=float)
    # momentum scheme
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)
    # regularization
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--self_distillation", default=0, type=float)
    parser.add_argument("--self_distillation_temperature", default=1, type=float)
    # configuration for different models.
    parser.add_argument("--densenet_growth_rate", default=12, type=int)
    parser.add_argument("--densenet_bc_mode", default=False, type=str2bool)
    parser.add_argument("--densenet_compression", default=0.5, type=float)
    parser.add_argument("--wideresnet_widen_factor", default=4, type=int)
    parser.add_argument("--rnn_n_hidden", default=200, type=int)
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--rnn_bptt_len", default=35, type=int)
    parser.add_argument("--rnn_clip", type=float, default=0.25)
    parser.add_argument("--rnn_use_pretrained_emb", type=str2bool, default=True)
    parser.add_argument("--rnn_tie_weights", type=str2bool, default=True)
    parser.add_argument("--rnn_weight_norm", type=str2bool, default=False)
    parser.add_argument("--transformer_n_layers", default=6, type=int)
    parser.add_argument("--transformer_n_head", default=8, type=int)
    parser.add_argument("--transformer_dim_model", default=512, type=int)
    parser.add_argument("--transformer_dim_inner_hidden", default=2048, type=int)
    parser.add_argument("--transformer_n_warmup_steps", default=4000, type=int)
    # miscs
    parser.add_argument("--same_seed_process", type=str2bool, default=True)
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    parser.add_argument("--summary_freq", default=256, type=int)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--track_time", default=False, type=str2bool)
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)
    # checkpoint
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=TRAINING_DIRECTORY,
        type=str,
        help="path to save checkpoint (default: checkpoint)",
    )
    parser.add_argument("--checkpoint_index", type=str, default=None)
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    parser.add_argument("--save_some_models", type=str, default=None, help="a list for comm_round to save")
    # device
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    parser.add_argument("--world", default=None, type=str, help="a list for devices.")
    parser.add_argument("--world_conf", default=None, type=str,
                        help="a list for the logic of world_conf follows a,b,c,d,e where: the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time); the block will be repeated for 'e' times.")
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--hostfile", type=str, default=None)
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    parser.add_argument("--mpi_env", type=str, default=None)
    """meta info."""
    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    parser.add_argument("--script_path", default="exp/", type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=1, type=int)
    """yaml"""
    parser.add_argument("--config_yaml", type=str, default="config.yaml")
    # parse conf.
    return parser


def parse_args_with_yaml(parser):
    args, unknown_args = parser.parse_known_args()
    # override default configurations with yaml file
    if args.config_yaml:
        with open(args.config_yaml, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args, unknown_args = parser.parse_known_args()

    key=None
    for item in unknown_args:
        if item.startswith('--'):
            if key:
                args.__setattr__(key, True)
            key = item[2:]
        else:
            value= item.replace('-', '_')
            if key:
                args.__setattr__(key, value)
                key = None
    return args


def save_config(args,yaml):
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(yaml) as f:
        f.write(args_text)
    return args_text


def complete_missing_config(conf):
    if "port" not in conf or conf.port is None:
        conf.port = get_free_port()
    if not conf.n_participated:
        conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    conf.timestamp = str(int(time.time()))


# find free port for communication.
def get_free_port():
    """ Get free port for communication."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def validate_config(conf):
    if conf.backend == "nccl" and conf.workers + 1 <= torch.cuda.device_count():
        raise ValueError("The NCCL backend requires exclusive access to CUDA devices.")


def init_config(conf):
    # define the graph for the computation.
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    set_random_seed(conf)

    set_device(conf)

    # init the model arch info.
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    dist.barrier()


def set_random_seed(conf):
    # init related to randomness on cpu.
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    # set seed to ensure experiment reproducibility.
    random.seed(conf.manual_seed)
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        traceback.print_exc()


def set_device(conf):
    if conf.on_cuda:
        # set cuda to ensure experiment reproducibility.
        if dist.get_backend() == "nccl":
            torch.cuda.set_device(torch.device("cuda:" + str(dist.get_rank())))
        else:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            available_memory = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                available_memory.append(mem_info.free)
            available_memory = np.array(available_memory)
            available_memory_patition = (available_memory / available_memory.sum()).cumsum()
            device_position = (dist.get_rank()) / dist.get_world_size()
            for i, patition in enumerate(available_memory_patition):
                if device_position <= patition:
                    break
            torch.cuda.set_device(torch.device("cuda:" + str(i)))
        # torch.cuda.set_device(torch.device("cuda:" + str(conf.graph.rank % torch.cuda.device_count())))
    else:
        torch.cuda.set_device(torch.device("cpu"))

def set_environ():
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['all_proxy'] = 'http://202.114.7.49:7890'
    os.environ['WANDB_MODE'] = 'online'

def experiment_paramenter(conf):
    if conf.data == 'music':
        conf.neighbor_sample_size = 8
        conf.dim = 16
        conf.n_iter = 1
        conf.weight_decay = 1e-5
        conf.lr = 5e-5
        conf.batch_size = 32
    elif conf.data == 'book':
        conf.neighbor_sample_size = 8
        conf.dim = 64
        conf.n_iter = 1
        conf.weight_decay = 2e-5
        conf.lr = 2e-4
        conf.batch_size = 32
    elif conf.data == 'movie':
        conf.neighbor_sample_size = 4
        conf.dim = 32
        conf.n_iter = 2
        conf.weight_decay = 1e-7
        conf.lr = 2e-2
        conf.batch_size = 32

# add debug environment
def debug_parameter(conf):
    # debug
    debug=False

    conf.data = 'movie'
    if debug==True:
        os.environ['WANDB_MODE'] = 'offline'
        conf.n_participated = 4
        conf.workers = 2
        conf.validation_interval = 1
        conf.topk_eval_interval = 1
    else:
        conf.n_participated = 32
        conf.workers = 32
        conf.validation_interval = 10
        conf.topk_eval_interval =30
    conf.train_fast = True
    conf.backend = "gloo"
    conf.n_comm_rounds = 2000*32
    conf.aggregator = "sum"
    conf.same_arch=True
    conf.experiment=f'fedKgcn_dataset_{conf.data}_np_{conf.n_participated}_nc_{conf.n_comm_rounds}'
    conf.k_list= [20, 50, 100]
    conf.local_batch_size = None
    return conf
