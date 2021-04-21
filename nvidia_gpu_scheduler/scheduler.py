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


class NVGPUScheduler_deprecated(object):
    def __init__(
        self,
        child_process,
        path_to_configs,
        child_args=lambda *_, **__: None,
        available_gpus=[0,1],
        config_fname_extension='',
        max_gpu_utilization=[100,30],
        max_jobs_per_gpu=[5,5],
        utilization_margin=5,
        time_between_tasks=60,
        child_verbose=False,
        **kwargs
        ):
        '''
        Task scheduler for NVIDIA GPUs

        args:
            :arg child_process: Child process
            :type child_process: function
            :arg path_to_configs: Directory path containing config files
            :type path_to_configs: str
            :arg child_args: Child process argument generator
            :type child_args: function
            :arg available_gpus: ID of available GPUs 
            :type available_gpus: list of ints
            :arg config_fname_extension: File extension of config files
            :type config_fname_extension: str
            :arg max_gpu_utilization: GPU utilization limit
            :type max_gpu_utilization: list of ints
            :arg max_jobs_per_gpu: Number of concurrent tasks limit
            :type max_jobs_per_gpu: list of ints
            :arg utilization_margin: GPU utilization margin (buffer)
            :type utilization_margin: int
            :arg time_between_tasks: Delay (in seconds) between tasks
            :type time_between_tasks: int
            :arg child_verbose: Child process verbose option
            :type child_verbose: bool

        returns:
        
        '''
        # input args
        assert len(available_gpus) == len(max_gpu_utilization) == len(max_jobs_per_gpu)
        assert os.path.isabs(path_to_configs) and os.path.isdir(path_to_configs)
        self.child_process = child_process
        self.path_to_configs = path_to_configs
        self.available_gpus = [int(elem) for elem in available_gpus]
        self.child_args = child_args
        self.config_ext = config_fname_extension
        self.max_gpu_utilization = [int(elem) for elem in max_gpu_utilization]
        self.max_jobs_per_gpu = [int(elem) for elem in max_jobs_per_gpu]
        self.utilization_margin = int(utilization_margin)
        self.time_between_tasks = int(time_between_tasks)
        self.child_verbose = child_verbose

        self.kwargs = kwargs

        # scheduler state
        self.resume = True
        self.terminate = False

        # prevent SIGINT signal from affecting the child processes
        signal.signal(signal.SIGINT, self._signal_handling)
        self.default_handler = signal.getsignal(signal.SIGINT)


    def _signal_handling(self, signum, frame):
        self.terminate = True
        print('pausing... Please wait!')


    def _get_child_process_args(self, f):
        output = self.child_args(f, **self.kwargs)
        return [] if output is None else [output]
    

    def run(self):
        # def run(args, subprocess_func, subprocess_verbose=False):

        list_of_configs = [abs_path for abs_path in sorted(glob.glob(self.path_to_configs + '/*%s'%(self.config_ext)))]
        list_of_gpus = self.available_gpus
        max_utilization = self.max_gpu_utilization
        max_jobs_per_gpu = self.max_jobs_per_gpu

        queued = copy.deepcopy(list_of_configs)
        if len(queued) == 0:
            raise AssertionError('No tasks(configs) given!')
        pools = {}
        running = OrderedDict()
        done = OrderedDict()
        failed = OrderedDict()
        curr_user = getpass.getuser()
        last_task_time = -float('inf')
        last_log_time = -float('inf')
        alpha = np.exp(-3/self.time_between_tasks)
        total_gpu_utilization_filt = [0.0]*len(list_of_gpus)
        user_gpu_utilization_filt = [0.0]*len(list_of_gpus)
        while len(queued) + len(running) > 0:
            time.sleep(0.01)

            # allocate GPU (every log_refresh_rate seconds)
            cand_gpu = []
            cand_gpu_util = []
            cand_gpumem_util = []
            if time.time() - last_log_time >= 1.0:
                total_compute_procs, user_compute_procs = get_num_procs(allocated_gpus=list_of_gpus, username=curr_user)
                total_gpu_utilization = get_gpu_utilization(allocated_gpus=list_of_gpus)
                total_gpumem_utilization = get_gpumem_utilization(allocated_gpus=list_of_gpus)
                user_gpu_utilization = [ceil(x/(y+1e-12)*z) for x, y, z in zip(user_compute_procs, total_compute_procs, total_gpu_utilization)]
                total_gpu_utilization_filt = [(1 - alpha)*x + alpha*X for x, X in zip(total_gpu_utilization, total_gpu_utilization_filt)]
                user_gpu_utilization_filt = [(1 - alpha)*x + alpha*X for x, X in zip(user_gpu_utilization, user_gpu_utilization_filt)]
                for i, gpuid, in enumerate(list_of_gpus):
                    tot_util_cond = total_gpu_utilization_filt[i] <= (100-self.utilization_margin)
                    tot_memutil_cond = total_gpumem_utilization[i] <= 50 # (1 - gpu_fraction)*100
                    user_util_cond = user_gpu_utilization_filt[i] < floor(max_utilization[i]*(100-self.utilization_margin)/100)
                    user_numproc_cond = user_compute_procs[i] < max_jobs_per_gpu[i] or max_jobs_per_gpu[i] == -1
                    if tot_util_cond and user_util_cond and user_numproc_cond and tot_memutil_cond:
                        cand_gpu.append(gpuid)
                        cand_gpu_util.append(total_gpu_utilization_filt[i])
                        cand_gpumem_util.append(total_gpumem_utilization[i])
            
            # run task (every time_between_tasks seconds)
            if len(queued) == 0 or len(cand_gpu) == 0 or time.time() - last_task_time < self.time_between_tasks: # no available GPUs or no queued tasks
                pass
            else:
                min_util_idx = cand_gpu_util.index(min(cand_gpu_util))
                if py3nvml.grab_gpus(num_gpus=1, gpu_select=[cand_gpu[min_util_idx]], gpu_fraction=0.5, max_procs=-1) == 0:
                    # if for some reason cannot allocate gpu
                    # print('CUDA_VISIBLE_DEVICES = %s'%(os.environ.get('CUDA_VISIBLE_DEVICES')))
                    last_task_time = time.time()
                    continue
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                if self.child_verbose:
                    p = multiprocessing.Pool(processes=1)
                else:
                    p = multiprocessing.Pool(processes=1, initializer=mute)
                pools[queued[0]] = p
                with open(queued[0], 'r') as f:
                    # running[queued[0]] = p.map_async(self.child_process, self._get_child_process_args(f))
                    running[queued[0]] = p.apply_async(self.child_process, self._get_child_process_args(f))
                signal.signal(signal.SIGINT, self.default_handler)
                queued.pop(0)
                last_task_time = time.time()
            
            # log (every log_refresh_rate seconds)
            if time.time() - last_log_time >= 1.0:
                # update thread status
                ready = []
                for key in running:
                    if running[key].ready(): # call has been executed
                        ready.append(key)
                        if running[key].successful(): # process terminated successfully
                            done[key] = running[key]
                        else: # process terminated with errors
                            failed[key] = running[key]
                for key in ready:
                    running.pop(key)
                    pools[key].close()
                    pools[key].terminate()
                    pools.pop(key)

                entry_len = 150
                print(''.center(entry_len,'+'))
                print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime(' %Y/%m/%d_%H:%M:%S ').center(entry_len,'-'))
                print(('+ USER: %s (process limit: %s, utilization limit: %s%%)'%(curr_user, max_jobs_per_gpu, max_utilization)).ljust(entry_len,' '))
                for i, gpuid in enumerate(list_of_gpus):
                    tup = (gpuid,)
                    tup += (user_compute_procs[i],)
                    tup += (total_compute_procs[i],)
                    tup += (user_gpu_utilization[i],)
                    tup += (total_gpu_utilization[i],)
                    tup += (total_gpumem_utilization[i],)
                    print(('+  gpu%d compute processes (%d/%d) utilization rate (%d%%/%d%%) memory usage (--%%/%d%%)'%tup).ljust(entry_len,' '))
                print((' %d QUEUED '%(len(queued))).center(entry_len,'-'))
                if self.kwargs.get('logging'):
                    print((' %d LOGGING '%(len(running))).center(entry_len,'-'))
                else:
                    print((' %d RUNNING '%(len(running))).center(entry_len,'-'))
                for key in running:
                    name_str = os.path.basename(key)
                    try:
                        tqdm_stat = pickle.load(open(os.path.join('/tmp', name_str + '.tqdm'), 'rb'))
                        tqdm_str = 'gpu%s pid=%d |%d%%| %d/%d [%s<%s, %sit/s]' % tqdm_stat
                    except:
                        tqdm_str = ''
                    name_str = '+  ' + name_str
                    print(name_str + tqdm_str.rjust(entry_len-len(name_str)))
                print((' %d FAILED '%(len(failed))).center(entry_len,'-'))
                for key in failed: print(os.path.basename(key))
                print((' %d DONE '%(len(done))).center(entry_len,'-'))
                for key in done: print(os.path.basename(key))
                print(''.center(entry_len,'+'))
                print('+')
                last_log_time = time.time()
        
            if self.terminate:
                self.resume = prompt_yes_or_no('Resume?')
                if self.resume:
                    IPython.embed()
                    self.terminate = False
            if self.terminate:
                break

        print('summary - done: %d, failed: %d, halted: %d, queued: %d' % (len(done), len(failed), len(running), len(queued)))


if __name__ == '__main__':

    scheduler = NVGPUScheduler(55555, 'hello')
    scheduler.start()
    scheduler.run(path_to_configs='/home/jgkim-larr/my_python_packages/nvidia-gpu-scheduler/example_configs',
        config_extension='.json'
    )
    