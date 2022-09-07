from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
import datetime
import dateutil.tz
import getpass
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
import time
from tqdm import tqdm
from types import SimpleNamespace

from nvidia_gpu_scheduler.utils import get_num_procs, get_gpu_utilization, get_gpumem_utilization
from nvidia_gpu_scheduler.utils import prompt_yes_or_no, mute_terminal as mute, ROSRate, get_random_string


class NVGPUWorker(SyncManager, ABC):

    def __init__(self, ip, port, authkey, name=None):

        # prevent SIGINT signal from affecting the manager
        signal.signal(signal.SIGINT, self._signal_handling)
        self.default_handler = signal.getsignal(signal.SIGINT)

        self.ip = ip
        self.port = port
        self.name = 'worker_%s'%(get_random_string(10)) if name is None else name
        super(NVGPUWorker, self).__init__(address=(self.ip, self.port), authkey=str.encode(authkey))
        print('Configured NVGPUWorker')

        print('Resource limits set to default profile:')
        self.set_limits()

        self.rate = ROSRate(1)

    def connect(self, *args, **kwargs):
        super(NVGPUWorker, self).connect(*args, **kwargs)
        print('NVGPUWorker connected to %s:%s'%(self.ip,self.port))

    def set_limits(
        self,
        available_gpus=[],
        gpu_utilization_limit=[],
        gpu_job_limit=[],
        utilization_margin=0,
        max_gpu_mem_usage=50,
        time_between_jobs=0,
        subprocess_verbose=False,
        apply_limits=['user', 'worker', 'all'][0]
        ):
        self.limits = SimpleNamespace()
        self.limits.available_gpus = available_gpus
        self.limits.gpu_utilization_limit = gpu_utilization_limit
        self.limits.gpu_job_limit = gpu_job_limit
        self.limits.utilization_margin = utilization_margin
        self.limits.max_gpu_mem_usage = max_gpu_mem_usage
        self.limits.time_between_jobs = time_between_jobs
        self.limits.subprocess_verbose = subprocess_verbose
        self.limits.apply_limits = apply_limits
        print('worker limits set to %s'%(self.limits))

    def update_limits(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.limits, key):
                setattr(self.limits, key, value)
        print('worker limits updated to %s'%(self.limits))
    
    def view_limits(self):
        print('self.limits = %s'%(self.limits))

    def run(self):

        # Access shared queues
        shared_pending_job_q = self.get_pending_job_q()
        shared_worker_status_q = self.get_worker_status_q()

        # Worker state
        self.worker_resume = True
        self.worker_terminate = False
        procs = {}
        running = OrderedDict()
        done = OrderedDict()
        failed = OrderedDict()
        last_job_time = -float('inf')

        alpha = np.exp(-3/self.limits.time_between_jobs)
        total_gpu_utilization_filt = {gpu_id: 0.0 for gpu_id in self.limits.available_gpus}
        user_gpu_utilization_filt = {gpu_id: 0.0 for gpu_id in self.limits.available_gpus}
        worker_gpu_utilization_filt = {gpu_id: 0.0 for gpu_id in self.limits.available_gpus}
        num_pending = shared_pending_job_q.qsize()
        while num_pending + len(running):
            curr_user = getpass.getuser()
            list_of_gpus = self.limits.available_gpus
            max_utilization = self.limits.gpu_utilization_limit
            max_jobs_per_gpu = self.limits.gpu_job_limit

            # 1. update candidate GPU
            total_compute_procs, user_compute_procs, pid_compute_procs = \
                get_num_procs(allocated_gpus=list_of_gpus, username=curr_user, version='v2')
            worker_compute_procs = copy.deepcopy(user_compute_procs)
            total_gpu_utilization = get_gpu_utilization(allocated_gpus=list_of_gpus)
            user_gpu_utilization = [ceil(x/(y+1e-12)*z) for x, y, z in zip(user_compute_procs, total_compute_procs, total_gpu_utilization)]
            total_gpumem_utilization, user_gpumem_utilization, pid_gpumem_utilization = \
                get_gpumem_utilization(allocated_gpus=list_of_gpus, username=curr_user, version='v2')
            
            total_gpu_utilization_filt = [(1 - alpha)*x + alpha*X for x, X in zip(total_gpu_utilization, total_gpu_utilization_filt)]
            user_gpu_utilization_filt = [(1 - alpha)*x + alpha*X for x, X in zip(user_gpu_utilization, user_gpu_utilization_filt)]

            cand_gpu, cand_gpu_util, cand_gpumem_util = [], [], []
            for i, gpuid, in enumerate(list_of_gpus):
                if gpuid < 0: # CPU mode
                    all_pid_compute_procs = [item for sublist in pid_compute_procs for item in sublist]
                    worker_compute_procs[i] = sum([running[key].pid not in all_pid_compute_procs for key in running])
                    user_compute_procs[i] = worker_compute_procs[i]
                else:
                    worker_compute_procs[i] = sum([running[key].pid in pid_compute_procs[i] for key in running])

                tot_util_cond = total_gpu_utilization_filt[i] <= (100-self.limits.utilization_margin)
                tot_memutil_cond = total_gpumem_utilization[i] <= self.limits.max_gpu_mem_usage # (1 - gpu_fraction)*100
                user_util_cond = user_gpu_utilization_filt[i] < floor(max_utilization[i]*(100-self.limits.utilization_margin)/100)
                user_numproc_cond = user_compute_procs[i] < max_jobs_per_gpu[i] or max_jobs_per_gpu[i] == -1
                worker_numproc_cond = worker_compute_procs[i] < max_jobs_per_gpu[i] or max_jobs_per_gpu[i] == -1
                tot_numproc_cond = total_compute_procs[i] < max_jobs_per_gpu[i] or max_jobs_per_gpu[i] == -1
                
                if self.limits.apply_limits == 'user':
                    is_cand = tot_util_cond and user_util_cond and user_numproc_cond and tot_memutil_cond 
                elif self.limits.apply_limits == 'worker':
                    is_cand = tot_util_cond and worker_numproc_cond and tot_memutil_cond
                elif self.limits.apply_limits == 'all':
                    is_cand = tot_util_cond and tot_numproc_cond and tot_memutil_cond
                else:
                    is_cand = False
                    print("Invalid apply_limits. Available options are ['user', 'worker']")

                if is_cand:
                    cand_gpu.append(gpuid)
                    cand_gpu_util.append(total_gpu_utilization_filt[i])
                    cand_gpumem_util.append(total_gpumem_utilization[i])

            # 2. run job process
            if len(cand_gpu) == 0 or time.time() - last_job_time < self.limits.time_between_jobs: # no available GPUs or no queued tasks
                pass
            else:
                min_util_idx = cand_gpu_util.index(min(cand_gpu_util))
                min_util_cand_gpu = cand_gpu[min_util_idx]
                if min_util_cand_gpu < 0: # CPU mode
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    grab_device_success = True
                else:
                    grab_device_success = py3nvml.grab_gpus(num_gpus=1, gpu_select=[cand_gpu[min_util_idx]], gpu_fraction=(100 - self.limits.max_gpu_mem_usage)/100, max_procs=-1, env_set_ok=True) > 0
                if not grab_device_success:
                    # if for some reason cannot allocate gpu
                    # print('CUDA_VISIBLE_DEVICES = %s'%(os.environ.get('CUDA_VISIBLE_DEVICES')))
                    # last_job_time = time.time()
                    continue
                try:
                    job = shared_pending_job_q.get_nowait() # {'tag': , 'config_byte': , 'worker_args': , 'worker_kwargs': }
                    num_pending -= 1
                    # {'tag': path, 'config': json.load(f, object_hook=lambda d : SimpleNamespace(**d)), 
                    # 'worker_args': worker_args, 'worker_kwargs': worker_kwargs}

                    signal.signal(signal.SIGINT, signal.SIG_IGN)

                    # recover namespace object if type is json
                    tmp_filepath = '/tmp/%s'%(job['tag'].replace('/','_'))
                    if os.path.splitext(tmp_filepath)[-1] == '.json':
                        with open(tmp_filepath,'wb') as f:
                            f.write(job['config_byte'])
                        with open(tmp_filepath,'r') as f:
                            job['config'] = json.load(f, object_hook=lambda d : SimpleNamespace(**d))
                        os.remove(tmp_filepath)
                        job['worker_kwargs'].update({'config': job['config'], 'config_byte': job['config_byte'], 'config_path': job['tag']})
                    else:
                        job['worker_kwargs'].update({'config_byte': job['config_byte'], 'config_path': job['tag']})
                    p = multiprocessing.Process(
                        target=self.worker,
                        args=job['worker_args'],
                        kwargs=job['worker_kwargs']
                    )
                    procs[job['tag']] = p
                    p.start()
                    running[job['tag']] = p
                    signal.signal(signal.SIGINT, self.default_handler)
                    
                    last_job_time = time.time()
                except queue.Empty:
                    pass
                except (EOFError, BrokenPipeError) as e:
                    print('lost connection to server')
            # update thread status
            ready = []
            for key in running:
                if not running[key].is_alive(): # call has been executed
                    ready.append(key)
                    if running[key].exitcode == 0: # process terminated successfully
                        done[key] = running[key]
                    else: # process terminated with errors
                        failed[key] = running[key]
            for key in ready:
                running.pop(key)
                procs[key].terminate()
                # procs[key].close()
                procs.pop(key)

            # 3. display status
            entry_len = 150
            print(''.center(entry_len,'+'))
            print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime(' %Y/%m/%d_%H:%M:%S ').center(entry_len,'-'))
            # worker status
            if self.limits.apply_limits == 'user':
                print('+ WORKER: %s (apply limits on user %s)'%(self.name, curr_user))
            elif self.limits.apply_limits == 'worker':
                print('+ WORKER: %s (apply limits on current worker)'%(self.name))
            elif self.limits.apply_limits == 'all':
                print('+ WORKER: %s (apply limits on all processes)'%(self.name))
            else:
                print("Invalid apply_limits. Available options are ['user', 'worker']")
            print(('+ (gpu_ids=%s, job_limit=%s, util_limit=%s%%)'%(list_of_gpus, max_jobs_per_gpu, max_utilization)).ljust(entry_len,' '))
            for i, gpuid in enumerate(list_of_gpus):
                tup = (gpuid,)
                tup += (user_compute_procs[i],)
                tup += (worker_compute_procs[i],)
                tup += (total_compute_procs[i],)
                tup += (user_gpu_utilization[i],)
                tup += (total_gpu_utilization[i],)
                tup += (user_gpumem_utilization[i],)
                tup += (total_gpumem_utilization[i],)
                print(('+  gpu%d compute processes (%d(%d)/%d) utilization rate (%d%%/%d%%) memory usage (%d%%/%d%%)'%tup).ljust(entry_len,' '))
            # job status
            print((' %d PENDING '%(num_pending)).center(entry_len,'-'))
            # if self.kwargs.get('logging'):
            #     print((' %d LOGGING '%(len(running))).center(entry_len,'-'))
            # else:
            #     print((' %d RUNNING '%(len(running))).center(entry_len,'-'))
            print((' %d LOGGING/RUNNING '%(len(running))).center(entry_len,'-'))
            tqdm_stats = []
            for key in running:
                name_str = os.path.basename(key)
                try:
                    tqdm_stat = pickle.load(open(os.path.join('/tmp', key.replace('/','_') + '.tqdm'), 'rb'))
                    tqdm_stats.append(tqdm_stat)
                    tqdm_str = 'gpu%s pid=%d |%d%%| %d/%d [%s<%s, %sit/s]' % tqdm_stat
                except:
                    tqdm_stats.append(None)
                    tqdm_str = ''
                name_str = '+  ' + name_str
                print(name_str + tqdm_str.rjust(entry_len-len(name_str)))
            print((' %d FAILED '%(len(failed))).center(entry_len,'-'))
            for key in failed: print(os.path.basename(key))
            print((' %d DONE '%(len(done))).center(entry_len,'-'))
            for key in done: print(os.path.basename(key))
            print(''.center(entry_len,'+'))
            print('+')

            # 4. report status to scheduler
            try:
                shared_worker_status_q.put({
                    self.name: {
                        'limit': vars(self.limits),
                        'status': {
                            'worker_compute_procs': user_compute_procs,
                            'total_compute_procs': total_compute_procs,
                            'worker_gpu_utilization': user_gpu_utilization,
                            'total_gpu_utilization': total_gpu_utilization,
                            'worker_gpumem_utilization': user_gpumem_utilization,
                            'total_gpumem_utilization': total_gpumem_utilization
                        },
                        'running': OrderedDict(((key, tqdm_stat) for key, tqdm_stat in zip(running, tqdm_stats))),
                        'done': OrderedDict(((key, None) for key in done)),
                        'failed': OrderedDict(((key, None) for key in failed)),
                        'last_updated': time.time()
                    }
                })
            except (EOFError, BrokenPipeError) as e: # lost connection to server
                print('lost connection to server')

            # 5. SIGINT(ctrl-c) handler
            if self.worker_terminate:
                self.worker_resume = prompt_yes_or_no('Resume?')
                if self.worker_resume:
                    IPython.embed()
                    self.worker_terminate = False
            if self.worker_terminate:
                for key in running:
                    running[key].terminate()
                break
            
            # run while loop every second
            self.rate.sleep()
            try: num_pending = shared_pending_job_q.qsize()
            except (EOFError, BrokenPipeError) as e: print('lost connection to server') # lost connection to server

        print('summary - done: %d, failed: %d, halted: %d, pending: %d' % (len(done), len(failed), len(running), num_pending))

    def _signal_handling(self, signum, frame):
        self.worker_terminate = True
        print('pausing worker... Please wait!')

    def worker(self, *args, **kwargs):
        if not self.limits.subprocess_verbose:
            mute()
            self.worker_function(*args, **kwargs)
        else:
            self.worker_function(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def worker_function(*args, config_path=None, config=None, **kwargs):
        pass
    
NVGPUWorker.register('get_pending_job_q')
NVGPUWorker.register('get_worker_status_q')


if __name__ == '__main__':

    class MyWorker(NVGPUWorker):
        @staticmethod
        def worker_function(*args, config_path=None, config=None, **kwargs):
            while True: time.sleep(1)

    worker = MyWorker('127.0.0.1', 55555, 'hello')
    worker.connect()
    worker.update_limits(
        available_gpus=[0,1],
        gpu_utilization_limit=[100,100],
        gpu_job_limit=[0,1],
        utilization_margin=0,
        time_between_jobs=30,
        subprocess_verbose=True,
        apply_limits='user'
    )

    worker.run()
