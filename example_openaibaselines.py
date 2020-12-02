import argparse
from pathlib import Path
from nvidia_gpu_scheduler.scheduler import NVGPUScheduler
from nvidia_gpu_scheduler.worker import NVGPUWorker
from nvidia_gpu_scheduler.utils import CatchExceptions
import sys
try:
    from baselines.run import main as run_baseline
except ImportError as e:
    print('This example requires OpenAI baselines (https://github.com/openai/baselines)')
    sys.exit()
try:
    import mujoco_py
except ImportError as e:
    print('This example requires mujoco-py (https://github.com/openai/mujoco-py)')
    sys.exit()


# def child_process(packaged_args):
#     from baselines.common.cmd_util import common_arg_parser
#     from nvidia_gpu_scheduler.utils import log_tqdm
#     import os
#     from tqdm import tqdm

#     args, config_dir = packaged_args # unpack
#     # Display addtional info (real-time child process progress not working)
#     arg_parser = common_arg_parser()
#     parsed_args, parsed_unknown_args = arg_parser.parse_known_args(args)
#     pbar = tqdm(total=parsed_args.num_timesteps)
#     config_filename = os.path.basename(config_dir)
#     log_tqdm(pbar, config_filename)
#     # Execute OpenAI baselines.run module
#     run_baseline(args)
#     # Close tqdm and remove tqdm logs
#     log_tqdm(pbar, config_filename, remove=True)
#     pbar.close()


# def child_process_args(f, **kwargs):
#     import json

#     params = json.load(f)
#     return params['sys.argv'], f.name


class OpenAIBaselinesExampleWorker(NVGPUWorker):
    @staticmethod
    def worker_function(*args, config_path=None, config=None, **kwargs):
        from baselines.common.cmd_util import common_arg_parser
        from nvidia_gpu_scheduler.utils import log_tqdm
        import os
        from tqdm import tqdm

        # Display addtional info (real-time child process progress not working)
        arg_parser = common_arg_parser()
        parsed_args, parsed_unknown_args = arg_parser.parse_known_args(*args)
        pbar = tqdm(total=parsed_args.num_timesteps)
        config_filename = os.path.basename(os.path.basename(config_path))
        log_tqdm(pbar, config_filename, remove=True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Execute OpenAI baselines.run module
        run_baseline(args)
        # Close tqdm and remove tqdm logs
        log_tqdm(pbar, config_filename, remove=True)
        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple example for NVIDIA GPU compute task scheduling utility (quad-GPU setup)')
    list_of_identities = ['scheduler', 'worker']
    parser.add_argument('--identity', type=str,
        help='Specify identity. Available identities are %s'%list_of_identities
    )
    args = parser.parse_args()
    assert args.identity in list_of_identities, 'Invalid identity. Available identities are %s'%list_of_identities
    
    if args.identity =='scheduler':
        scheduler = NVGPUScheduler(55555, 'openaibaselines_example')
        scheduler.start()
        scheduler.run(path_to_configs=str((Path(__file__).parent / 'example_openaibaselines_configs').resolve()),
            config_extension='.json'
        )
    elif args.identity == 'worker':
        worker = OpenAIBaselinesExampleWorker('127.0.0.1', 55555, 'openaibaselines_example', name='local')
        worker.connect()
        worker.update_limits(
            available_gpus=[0,1,2,3],
            gpu_utilization_limit=[100,100,100,100],
            gpu_job_limit=[3,3,1,1],
            utilization_margin=5,
            time_between_jobs=3,
            subprocess_verbose=False,
            apply_limits='user'
        )
        worker.run()

# DEPRECATED
# if __name__ == '__main__':
#     from nvidia_gpu_scheduler import NVGPUScheduler_deprecated
#     '''
#     OpenAI baselines example
    
#     OpenAI baselines (https://github.com/openai/baselines) and its dependencies (mujoco-py, ...) are required to run this example!
#     Tracking child process progress was NOT implemented for this example. (requires modifying the OpenAI baselines source code or some hacky workaround)

#     '''
#     parser = argparse.ArgumentParser(description='Simple example for NVIDIA GPU compute task scheduling utility (quad-GPU setup)')
#     parser.add_argument('--available_gpus', nargs='+', type=int, default=[0,1,2,3],
#         help='List of available GPU(s)'
#     )
#     parser.add_argument('--config_fname_extension', type=str, default='.json',
#         help='File extension of config file(s)'
#     )
#     parser.add_argument('--max_gpu_utilization', nargs='+', type=int, default=[100,100,30,30],
#         help='Maximum utilization rate for each GPU'
#     )
#     parser.add_argument('--max_jobs_per_gpu', nargs='+', type=int, default=[3,3,1,1],
#         help='Maximum number of concurrent jobs'
#     )
#     parser.add_argument('--utilization_margin', type=int, default=5,
#         help='Percent margin for maximum GPU utilization rate'
#     )
#     parser.add_argument('--time_between_tasks', type=int, default=15,
#         help='Time delay in seconds between tasks'
#     )
#     parser.add_argument('--child_verbose', action='store_true',
#         help='Allow child process(s) to output to terminal'
#     )
#     parser.add_argument('--logging', action='store_true',
#         help='Enable logging for child process(s)'
#     )
#     args = parser.parse_args()
    
#     path_to_configs = str((Path(__file__).parent / 'example_openaibaselines_configs').resolve())
#     # 1. child process does NOT display traceback upon uncaught exception by default
#     manager1 = NVGPUScheduler_deprecated(child_process, path_to_configs, child_args=child_process_args, **vars(args))
#     manager1.run()
#     # 2. use CatchExceptions wrapper to display child process traceback upon uncaught exception
#     manager2 = NVGPUScheduler_deprecated(CatchExceptions(child_process), path_to_configs, child_args=child_process_args, **vars(args))
#     manager2.run()