import argparse
from pathlib import Path
from nvidia_gpu_scheduler.scheduler import NVGPUScheduler
try: from baselines.run import main as run_baseline
except ImportError as e: print('This example requires OpenAI baselines (https://github.com/openai/baselines)')
try: import mujoco_py
except ImportError as e: print('This example requires mujoco-py (https://github.com/openai/mujoco-py)')


def child_process(packaged_args):
    from baselines.common.cmd_util import common_arg_parser
    from nvidia_gpu_scheduler.utils import log_tqdm
    import os
    from tqdm import tqdm

    args, config_dir = packaged_args # unpack
    # Display addtional info (real-time child process progress not working)
    arg_parser = common_arg_parser()
    parsed_args, parsed_unknown_args = arg_parser.parse_known_args(args)
    pbar = tqdm(total=parsed_args.num_timesteps)
    config_filename = os.path.basename(config_dir)
    log_tqdm(pbar, config_filename)
    # Execute OpenAI baselines.run module
    run_baseline(args)
    # Close tqdm and remove tqdm logs
    log_tqdm(pbar, config_filename, remove=True)
    pbar.close()


def child_process_args(f, **kwargs):
    import json

    params = json.load(f)
    return params['sys.argv'], f.name


if __name__ == '__main__':
    '''
    OpenAI baselines example
    
    OpenAI baselines (https://github.com/openai/baselines) and its dependencies (mujoco-py, ...) are required to run this example!
    Tracking child process progress was NOT implemented for this example. (requires modifying the OpenAI baselines source code or some hacky workaround)

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
    parser.add_argument('--time_between_tasks', type=int, default=15,
        help='Time delay in seconds between tasks'
    )
    parser.add_argument('--child_verbose', action='store_true',
        help='Allow child process(s) to output to terminal'
    )
    parser.add_argument('--logging', action='store_true',
        help='Enable logging for child process(s)'
    )
    args = parser.parse_args()
    
    path_to_configs = str((Path(__file__).parent / 'example_openaibaselines_configs').resolve())
    manager = NVGPUScheduler(child_process, path_to_configs, child_args=child_process_args, **vars(args))
    manager.run()
