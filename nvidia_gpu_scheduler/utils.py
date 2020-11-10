import datetime
import logging
import numpy as np
import os
import pickle
import pwd
import py3nvml
import sys
import time
from tqdm import tqdm
import warnings


class ROSRate(object):
    '''
    http://docs.ros.org/diamondback/api/rostime/html/rate_8cpp_source.html
    '''
    def __init__(self, frequency):
        assert frequency > 0, 'Frequency must be greated than zero!'
        self._freq = frequency
        self._start = time.time()
        self._actual_cycle_time = 1/self._freq

    def reset(self):
        self._start = time.time()
    
    def sleep(self):
        expected_end = self._start + 1/self._freq
        actual_end = time.time()

        if actual_end < self._start: # detect backward jumps in time
            expected_end = actual_end + 1/self._freq

        # calculate sleep time
        sleep_duration = expected_end - actual_end
        # set the actual amount of time the loop took in case the user wants to know
        self._actual_cycle_time = actual_end - self._start

        # reset start time
        self._start = expected_end

        if sleep_duration <= 0:
            # if we've jumped forward in time, or the loop has taken more than a full extra cycle, reset our cycle
            if actual_end > expected_end + 1/self._freq:
                self._start = actual_end
            return True

        return time.sleep(sleep_duration)


def mute_terminal():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def prompt_yes_or_no(query):
    while True:
        response = input(query + ' [Y/n] ').lower()
        if response in {'y', 'yes'}:
            return True
        elif response in {'n', 'no'}:
            return False
        else:
            print('Invalid response!\n')


def log_tqdm(tqdm_obj, config_fname, remove=False):
    if remove:
        try: os.remove(os.path.join('/tmp', config_fname + '.tqdm'))
        except OSError: pass
    else:
        d = tqdm_obj.format_dict
        tqdm_stat = ()
        tqdm_stat += (os.environ.get('CUDA_VISIBLE_DEVICES'),)
        tqdm_stat += (os.getpid(),)
        tqdm_stat += (int(d['n']/d['total']*100),)
        tqdm_stat += (d['n'],)
        tqdm_stat += (d['total'],)
        tqdm_stat += (str(datetime.timedelta(seconds=int(d['elapsed']))),)
        try: tqdm_stat += (str(datetime.timedelta(seconds=int((d['total'] - d['n'])/d['rate']))),)
        except: tqdm_stat += ('?',)
        try: tqdm_stat += (round(d['rate'],2),)
        except: tqdm_stat += ('?',)
        pickle.dump(tqdm_stat,
            open(os.path.join('/tmp', config_fname + '.tqdm'), 'wb')
        )


def get_num_procs(allocated_gpus=[], username='all users'):
    """ Gets the number of processes running on each gpu

    Returns
    -------
    num_procs : list(int)
        Number of processes running on each gpu

    Note
    ----
    If function can't query the driver will return an empty list rather than raise an
    Exception.

    Note
    ----
    If function can't get the info from the gpu will return -1 in that gpu's place
    """
    if username != 'all users': pwd.getpwnam(username)

    # Try connect with NVIDIA drivers
    logger = logging.getLogger(__name__)
    try:
        py3nvml.py3nvml.nvmlInit()
    except:
        str_ = """Couldn't connect to nvml drivers. Check they are installed correctly."""
        warnings.warn(str_, RuntimeWarning)
        logger.warn(str_)
        return [], []

    num_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()
    if len(allocated_gpus) == 0:
        allocated_gpus = list(range(num_gpus))
    else:
        assert num_gpus > max(allocated_gpus)
    gpu_procs = [-1]*len(allocated_gpus)
    gpu_procs_user = [-1]*len(allocated_gpus)
    for i, gpuid in enumerate(allocated_gpus):
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        procs = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetComputeRunningProcesses, h,
                             ['something'])
        gpu_procs[i] = len(procs)
        procs_user = []
        for proc in procs:
            proc_uid = os.stat('/proc/%d' % (proc.pid)).st_uid
            if pwd.getpwuid(proc_uid).pw_name == username or username == 'all users':
                procs_user.append(proc)
        gpu_procs_user[i] = len(procs_user)

    py3nvml.py3nvml.nvmlShutdown()
    return gpu_procs, gpu_procs_user


def get_gpu_utilization(allocated_gpus=[]):
    '''
    Gets the utilization rates of each gpu
    '''
    # Try connect with NVIDIA drivers
    logger = logging.getLogger(__name__)
    try:
        py3nvml.py3nvml.nvmlInit()
    except:
        str_ = """Couldn't connect to nvml drivers. Check they are installed correctly."""
        warnings.warn(str_, RuntimeWarning)
        logger.warn(str_)
        return []

    num_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()
    if len(allocated_gpus) == 0:
        allocated_gpus = list(range(num_gpus))
    else:
        assert num_gpus > max(allocated_gpus)
    gpu_rates = [-1]*len(allocated_gpus)
    for i, gpuid in enumerate(allocated_gpus):
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        rate = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetUtilizationRates, h,
                             ['something'])
        gpu_rates[i] = rate.gpu

    py3nvml.py3nvml.nvmlShutdown()
    return gpu_rates


def get_gpumem_utilization(allocated_gpus=[]):
    '''
    Gets the memory usage of each gpu
    '''
    # Try connect with NVIDIA drivers
    logger = logging.getLogger(__name__)
    try:
        py3nvml.py3nvml.nvmlInit()
    except:
        str_ = """Couldn't connect to nvml drivers. Check they are installed correctly."""
        warnings.warn(str_, RuntimeWarning)
        logger.warn(str_)
        return []

    num_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()
    if len(allocated_gpus) == 0:
        allocated_gpus = list(range(num_gpus))
    else:
        assert num_gpus > max(allocated_gpus)
    mem_rates = [-1]*len(allocated_gpus)
    for i, gpuid in enumerate(allocated_gpus):
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        info = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetMemoryInfo, h,
                             ['something'])
        mem_rates[i] = int(np.ceil(100*info.used/info.total))

    py3nvml.py3nvml.nvmlShutdown()
    return mem_rates