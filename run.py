import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import traceback
from distutils import dist
import torch.multiprocessing as mp
from parameters import get_args, init_config
from pcode.master import Master
from pcode.utils.auto_distributed import *
from pcode.worker import Worker


# -*- coding: utf-8 -*-


def run(conf):
    process = Master(conf) if conf.graph.rank == 0 else Worker(conf)
    process.run()


def init_process(rank, size, fn, conf):
    # init the distributed world.
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = conf.port
        dist.init_process_group(conf.backend, rank=rank, world_size=size)
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False
    try:
        warnings.filterwarnings("ignore", category=UserWarning)

        # init the config.
        init_config(conf)

        # start federated learning.
        fn(conf)

    except Exception as e:
        print(f"Caught exception in rank {rank}")
        traceback.print_exc()
        raise e


def is_mpi_enabled():
    return 'MPI_COMM_WORLD_SIZE' in os.environ


def set_working_directory():
    current_file = os.path.abspath(__file__)
    directory = os.path.dirname(current_file)
    os.chdir(directory)


def run_mpi():
    if is_mpi_enabled():
        init_process(0, 0, run, conf)
    else:
        os.environ['MPI_COMM_WORLD_SIZE'] = size.__str__()
        args_str = ' '.join(sys.argv[1:])
        python_prefix = sys.prefix
        os.system(
            f'$HOME/.openmpi/bin/mpirun -n {size} --mca orte_base_help_aggregate 0 --mca btl_tcp_if_exclude docker0,lo --hostfile {conf.hostfile} {python_prefix}/bin/python run_gloo.py ' + args_str)


def run_gloo():
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, conf))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    # get config.

    conf = get_args()

    set_working_directory()

    # Create process for each worker and master.
    size = conf.workers + 1

    mp.set_start_method("spawn")

    if conf.backend == 'mpi':
        run_mpi()

    elif conf.backend in ['gloo','nccl']:
        run_gloo()
