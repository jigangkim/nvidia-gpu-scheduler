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


class OpenAIBaselinesExampleWorker(NVGPUWorker):
    @staticmethod
    @CatchExceptions
    def worker_function(*args, config_path=None, config=None, config_byte=None, **kwargs):
        from baselines.common.cmd_util import common_arg_parser
        from nvidia_gpu_scheduler.utils import log_tqdm
        import os
        from tqdm import tqdm

        # Display addtional info (real-time child process progress not working)
        arg_parser = common_arg_parser()
        parsed_args, parsed_unknown_args = arg_parser.parse_known_args(*args)
        pbar = tqdm(total=parsed_args.num_timesteps)
        log_tqdm(pbar, config_path.replace('/','_'), remove=False)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Execute OpenAI baselines.run module (pbar not updated)
        run_baseline(args)
        # Close tqdm and remove tqdm logs
        log_tqdm(pbar, config_path.replace('/','_'), remove=True)
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