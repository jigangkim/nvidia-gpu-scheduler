from collections import OrderedDict
import copy
import datetime
import dateutil.tz
import getpass
import glob
import IPython
import json
from math import ceil, floor
import numpy as np
import multiprocessing
# multiprocessing.set_start_method('spawn', True) # hacky workaround for ptvsd (python debugger for vscode)
from multiprocessing.managers import SyncManager
import os
import pickle
import py3nvml
import queue
import signal
import socket
import time
from tqdm import tqdm
from types import SimpleNamespace
import urllib.request

from nvidia_gpu_scheduler.utils import get_num_procs, get_gpu_utilization, get_gpumem_utilization
from nvidia_gpu_scheduler.utils import prompt_yes_or_no, mute_terminal as mute, ROSRate


class NVGPUScheduler(SyncManager):
    
    counter = 0
    pending_job_q = queue.Queue() # scheduler -> worker(s)
    worker_status_q = queue.Queue() # worker(s) -> scheduler

    def __init__(self, port, authkey, ip_type='local'):

        NVGPUScheduler.counter += 1
        if NVGPUScheduler.counter > 1:
            raise Exception('An instance of NVGPUScheduler already exist! (existing instances: %d)'%(NVGPUScheduler.counter))

        # prevent SIGINT signal from affecting the manager
        signal.signal(signal.SIGINT, self._signal_handling)
        self.default_handler = signal.getsignal(signal.SIGINT)

        if ip_type == 'public':
            self.ip = urllib.request.urlopen('https://api.ipify.org/').read().decode('utf8')
        elif ip_type == 'primary': # either public or (if behind NAT) private 
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('1.1.1.1', 53))
            self.ip = s.getsockname()[0]
            s.close()
        elif ip_type == 'local':
            self.ip = '127.0.0.1'
        else:
            raise ValueError('Invalid ip_type')
        self.port = port
        super(NVGPUScheduler, self).__init__(address=(self.ip, self.port), authkey=str.encode(authkey))
        print('Configured NVGPUScheduler')

        self.rate = ROSRate(1)

    def start(self, *args, **kwargs):
        super(NVGPUScheduler, self).start(*args, **kwargs)
        print('NVGPUScheduler opened at %s:%s'%(self.ip,self.port))

    def run(
        self,
        *worker_args,
        path_to_configs='',
        config_extension='',
        **worker_kwargs
        ):
        '''
        Scheduler for NVIDIA GPUs

        args:
            :arg worker_args: Arguments to be passed along to the worker
            :type worker_args: tuple
            :arg path_to_configs: Path containing config files
            :type path_to_configs: list of strings
            :arg config_extension: File extension of config files
            :type config_extension: str
            :arg worker_kwargs: Keyword arguments to be passed along to the 
            :type worker_kwargs: dict

        '''

        # Start a shared manager server and access its queues
        shared_pending_job_q = self.get_pending_job_q()
        shared_worker_status_q = self.get_worker_status_q()

        # Scheduler state
        self.scheduler_resume = True
        self.scheduler_terminate = False

        list_of_configs = [os.path.abspath(match) for match in sorted(glob.glob(path_to_configs + '/*%s'%(config_extension)))]
        assert len(list_of_configs) > 0, 'No configs available!'
        print('Found %d configs'%(len(list_of_configs)))
        for path in list_of_configs:
            with open(path, 'rb') as f:
                shared_pending_job_q.put({'tag': path, 'config_byte': f.read(), 
                    'worker_args': worker_args, 'worker_kwargs': worker_kwargs}
                )

        worker_status = OrderedDict()

        self.rate.reset()

        num_pending = shared_pending_job_q.qsize()
        while num_pending + sum([len(x['running']) for _, x in worker_status.items()]) > 0:
            # 1. update worker_status
            try:
                while True:
                    outdict = shared_worker_status_q.get_nowait()
                    for name, status in outdict.items(): status['last_updated'] = time.time()
                    worker_status.update(outdict)
            except queue.Empty:
                pass
            num_pending = shared_pending_job_q.qsize()
            
            # 2. display worker_status
            entry_len = 150
            print(''.center(entry_len,'+'))
            print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime(' %Y/%m/%d_%H:%M:%S ').center(entry_len,'-'))
            print('+ SCHEDULER: %d worker(s) connected'%(len(worker_status)))
            # worker status
            for name, status in worker_status.items():
                gpu_ids = status['limit']['available_gpus']
                job_limit = status['limit']['gpu_job_limit']
                util_limit = status['limit']['gpu_utilization_limit']
                last_updated_seconds_ago = time.time() - status['last_updated']
                print(('+ (worker=%s, gpu_ids=%s, job_limit=%s, util_limit=%s, last_updated=%ds ago)'%(name, gpu_ids, job_limit, util_limit, last_updated_seconds_ago)).ljust(entry_len,' '))
                worker_compute_procs = status['status']['worker_compute_procs']
                total_compute_procs = status['status']['total_compute_procs']
                worker_gpu_utilization = status['status']['worker_gpu_utilization']
                total_gpu_utilization = status['status']['total_gpu_utilization']
                worker_gpumem_utilization = status['status']['worker_gpumem_utilization']
                total_gpumem_utilization = status['status']['total_gpumem_utilization']
                for i, gpu_id in enumerate(gpu_ids):
                    tup = (gpu_id,)
                    tup += (worker_compute_procs[i],)
                    tup += (total_compute_procs[i],)
                    tup += (worker_gpu_utilization[i],)
                    tup += (total_gpu_utilization[i],)
                    tup += (worker_gpumem_utilization[i],)
                    tup += (total_gpumem_utilization[i],)
                    print(('+  gpu%d compute processes (%d/%d) utilization rate (%d%%/%d%%) memory usage (%d%%/%d%%)'%tup).ljust(entry_len,' '))
            # job status
            print((' %d PENDING '%(num_pending)).center(entry_len,'-'))
            num_running = sum([len(status['running']) for name, status in worker_status.items()])
            if worker_kwargs.get('logging'): print((' %d LOGGING '%(num_running)).center(entry_len,'-'))
            else: print((' %d RUNNING '%(num_running)).center(entry_len,'-'))
            for name, status in worker_status.items():
                running = status['running']
                for config_name, tqdm_stat in running.items():
                    name_str = '+  ' + os.path.basename(config_name)
                    try:
                        status_str = '%s '%(name) + 'gpu%s pid=%d |%d%%| %d/%d [%s<%s, %sit/s]'%tqdm_stat
                    except:
                        status_str = name
                    print(name_str + status_str.rjust(entry_len-len(name_str)))
            num_failed = sum([len(status['failed']) for name, status in worker_status.items()])
            print((' %d FAILED '%(num_failed)).center(entry_len,'-'))
            for name, status in worker_status.items():
                failed = status['failed']
                for config_name, _ in failed.items():
                    name_str = os.path.basename(config_name)
                    print(name_str + name.rjust(entry_len-len(name_str)))
            num_done = sum([len(status['done']) for name, status in worker_status.items()])
            print((' %d DONE '%(num_done)).center(entry_len,'-'))
            for name, status in worker_status.items():
                done = status['done']
                for config_name, _ in done.items():
                    name_str = os.path.basename(config_name)
                    print(name_str + name.rjust(entry_len-len(name_str)))
            print(''.center(entry_len,'+'))
            print('+')

            # 3. exception handling for unresponsive worker(s)
            unresponsive_workers = []
            for name, status in worker_status.items():
                if time.time() - status['last_updated'] > 600:
                    print('worker %s has been unresponsive for 10 min!'%(name))
                    unresponsive_workers.append(name)
                    unresponsive_running = status['running']
                    for path in unresponsive_running:
                        # set unresponsive running jobs back to pending jobs
                        with open(path, 'rb') as f:
                            shared_pending_job_q.put({'tag': path, 'config_byte': f.read(), 
                                'worker_args': worker_args, 'worker_kwargs': worker_kwargs}
                            )
            for name in unresponsive_workers:
                worker_status.pop(name)
            num_pending = shared_pending_job_q.qsize()

            # 4. SIGINT(ctrl-c) handler
            if self.scheduler_terminate:
                self.scheduler_resume = prompt_yes_or_no('Resume?')
                if self.scheduler_resume:
                    IPython.embed()
                    self.scheduler_terminate = False
            if self.scheduler_terminate:
                break

            # run while loop every second
            self.rate.sleep()

        time.sleep(2)
        self.shutdown()

    def _signal_handling(self, signum, frame):
        self.scheduler_terminate = True
        print('pausing scheduler... Please wait!')

    def __del__(self):
        NVGPUScheduler.counter -= 1

NVGPUScheduler.register('get_pending_job_q', callable=lambda: NVGPUScheduler.pending_job_q)
NVGPUScheduler.register('get_worker_status_q', callable=lambda: NVGPUScheduler.worker_status_q)


if __name__ == '__main__':

    scheduler = NVGPUScheduler(55555, 'hello')
    scheduler.start()
    scheduler.run(path_to_configs='/home/jgkim-larr/my_python_packages/nvidia-gpu-scheduler/example_configs',
        config_extension='.json'
    )
    