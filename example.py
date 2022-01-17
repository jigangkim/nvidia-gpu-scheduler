import argparse
import gin
from pathlib import Path
from nvidia_gpu_scheduler.scheduler import NVGPUScheduler
from nvidia_gpu_scheduler.worker import NVGPUWorker
from nvidia_gpu_scheduler.utils import CatchExceptions


class ExampleWorkerJSON(NVGPUWorker):
    @staticmethod
    @CatchExceptions
    def worker_function(*args, config_path=None, config=None, **kwargs):
        import os
        import random
        import time
        from tqdm import tqdm
        from nvidia_gpu_scheduler.utils import log_tqdm

        n = config.steps
        pbar = tqdm(total=n)
        log_tqdm(pbar, config_path.replace('/','_'))
        for i in range(n):
            time.sleep(1)
            pbar.update(n=1)
            log_tqdm(pbar, config_path.replace('/','_'))
        log_tqdm(pbar, config_path.replace('/','_'), remove=True)
        pbar.close()
        # intentionally trigger with 50% probability
        trigger_error = random.random() < config.error_rate
        if trigger_error:
            raise ValueError('failed with 50% probability')
        else:
            print('succeeded with 50% probability')


class ExampleWorkerGIN(NVGPUWorker):
    @staticmethod
    @CatchExceptions
    def worker_function(*args, config_path=None, config_byte=None, **kwargs):
        import os
        import random
        import time
        from tqdm import tqdm
        from nvidia_gpu_scheduler.utils import log_tqdm

        @gin.configurable
        def run(steps=0, error_rate=0):
            pbar = tqdm(total=steps)
            log_tqdm(pbar, config_path.replace('/','_'))
            for i in range(steps):
                time.sleep(1)
                pbar.update(n=1)
                log_tqdm(pbar, config_path.replace('/','_'))
            log_tqdm(pbar, config_path.replace('/','_'), remove=True)
            pbar.close()
            # intentionally trigger error
            trigger_error = random.random() < error_rate
            if trigger_error:
                raise ValueError('failed with %d%% probability'%(error_rate*100))
            else:
                print('succeeded with %d%% probability'%(100-error_rate*100))

        tmp_gin_file = '/tmp/%s'%(config_path.replace('/','_'))
        with open(tmp_gin_file,'wb') as f:
            f.write(config_byte)
        gin.parse_config_file(tmp_gin_file)
        os.remove(tmp_gin_file)
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple example for NVIDIA GPU compute task scheduling utility (quad-GPU setup)')
    list_of_identities = ['scheduler', 'worker']
    list_of_config_exts = ['.json', '.gin']
    parser.add_argument('--identity', type=str,
        help='Specify identity. Available identities are %s'%list_of_identities
    )
    parser.add_argument('--config_ext', type=str, default='.json',
        help='Specify config extension. Available types are %s'%list_of_config_exts
    )
    args = parser.parse_args()
    assert args.identity in list_of_identities, 'Invalid identity. Available identities are %s'%list_of_identities
    assert args.config_ext in list_of_config_exts, 'Invalid config extension. Available types are %s'%list_of_config_exts
    if args.config_ext == '.json':
        worker_class = ExampleWorkerJSON
    elif args.config_ext == '.gin':
        worker_class = ExampleWorkerGIN
    
    if args.identity =='scheduler':
        scheduler = NVGPUScheduler(55555, 'simple_example')
        scheduler.start()
        scheduler.run(path_to_configs=str((Path(__file__).parent / 'example_configs').resolve()),
            config_extension=args.config_ext
        )
    elif args.identity == 'worker':
        worker = worker_class('127.0.0.1', 55555, 'simple_example', name='local')
        worker.connect()
        worker.update_limits(
            available_gpus=[-1,0,1,2,3],
            gpu_utilization_limit=[100,100,100,100,100],
            gpu_job_limit=[3,0,0,0,0],
            utilization_margin=5,
            time_between_jobs=3,
            subprocess_verbose=False,
            apply_limits='user'
        )
        worker.run()