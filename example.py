import argparse
from pathlib import Path
from nvidia_gpu_scheduler.scheduler import NVGPUScheduler_deprecated as NVGPUScheduler, CatchExceptions


def child_process(args):
    import os
    import random
    import time
    from tqdm import tqdm
    from nvidia_gpu_scheduler.utils import log_tqdm

    n = 10
    pbar = tqdm(total=n)
    config_filename = os.path.basename(args.config_dir)
    log_tqdm(pbar, config_filename)
    for i in range(n):
        time.sleep(1)
        pbar.update(n=1)
        log_tqdm(pbar, config_filename)
    log_tqdm(pbar, config_filename, remove=True)
    pbar.close()
    # intentionally trigger with 50% probability
    trigger_error = random.choice([True, False])
    if trigger_error:
        raise ValueError('failed with 50% probability')
    else:
        print('succeeded with 50% probability')


def child_process_args(f, logging=False, **kwargs):
    import json
    from types import SimpleNamespace

    # reconstruct arguments
    params = json.load(f, object_hook=lambda d : SimpleNamespace(**d)) # read json as namespace
    args = SimpleNamespace()
    if hasattr(params, 'ddpg'):
        args.config_type = 'ddpg'
    elif hasattr(params, 'lnt'):
        args.config_type = 'lnt-ddpg'
    else:
        raise ValueError('Cannot find CONFIG_TYPE!')
    args.config_dir = f.name
    args.absolute_path = True
    args.logging = logging
    args.render = False
    args.resume = False
    args.playback = False
    return args


if __name__ == '__main__':
    '''
    Dummy example
    
    User should provide child process function and child process argument function.
    Incorporating nvidia_gpu_scheduler.utils.log_tqdm into the chlid process function is recommended to track child process progress.

    '''
    parser = argparse.ArgumentParser(description='Simple example for NVIDIA GPU compute task scheduling utility (quad-GPU setup)')
    parser.add_argument('--available_gpus', nargs='+', type=int, default=[0,1,2,3],
        help='List of available GPU(s)'
    )
    parser.add_argument('--config_fname_extension', type=str, default='.json',
        help='File extension of config file(s)'
    )
    parser.add_argument('--max_gpu_utilization', nargs='+', type=int, default=[100,100,30,30],
        help='Maximum utilization rate for each GPU'
    )
    parser.add_argument('--max_jobs_per_gpu', nargs='+', type=int, default=[3,3,1,1],
        help='Maximum number of concurrent jobs'
    )
    parser.add_argument('--utilization_margin', type=int, default=5,
        help='Percent margin for maximum GPU utilization rate'
    )
    parser.add_argument('--time_between_tasks', type=int, default=3,
        help='Time delay in seconds between tasks'
    )
    parser.add_argument('--child_verbose', action='store_false',
        help='Allow child process(s) to output to terminal'
    )
    parser.add_argument('--logging', action='store_true',
        help='Enable logging for child process(s)'
    )
    args = parser.parse_args()
    
    path_to_configs = str((Path(__file__).parent / 'example_configs').resolve())
    # 1. child process does NOT display traceback upon uncaught exception by default
    manager1 = NVGPUScheduler(child_process, path_to_configs, child_args=child_process_args, **vars(args))
    manager1.run()
    # 2. use CatchExceptions wrapper to display child process traceback upon uncaught exception
    manager2 = NVGPUScheduler(CatchExceptions(child_process), path_to_configs, child_args=child_process_args, **vars(args))
    manager2.run()
