import datetime
import errno
import logging
import numpy as np
import os
import pickle
import pwd
import py3nvml
import random
import string
import sys
import time
from tqdm import tqdm
import traceback
import warnings


# https://stackoverflow.com/questions/6728236/exception-thrown-in-multiprocessing-pool-not-detected
class CatchExceptions(object):
    '''
    Wrapper for callable enabling child process exception/traceback catching
    '''
    def __init__(self, callable):
        self.__callable = callable


    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            logging.getLogger().critical('Uncaught error', exc_info=sys.exc_info())
            raise
        return result


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
        try:
            pickle.dump(tqdm_stat,
                open(os.path.join('/tmp', config_fname + '.tqdm'), 'wb')
            )
        except OSError as e:
            if e.errno == errno.ENOENT: print('log_tqdm: No such file of directory')
            elif e.errno == errno.ENOSPC: print('log_tqdm: No space left on device')


def get_random_string(length):
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def get_num_procs(allocated_gpus=[], username='all users', version='v1'):
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
    gpu_procs_pid = [[]]*len(allocated_gpus)
    for i, gpuid in enumerate(allocated_gpus):
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        procs = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetComputeRunningProcesses, h,
                             ['something'])
        gpu_procs[i] = len(procs)
        procs_user = []
        procs_pid = []
        for proc in procs:
            try:
                proc_uid = os.stat('/proc/%d' % (proc.pid)).st_uid
                if pwd.getpwuid(proc_uid).pw_name == username or username == 'all users':
                    procs_user.append(proc)
            except:
                pass
            procs_pid.append(proc.pid)
        gpu_procs_user[i] = len(procs_user)
        gpu_procs_pid[i] = procs_pid

    py3nvml.py3nvml.nvmlShutdown()

    if version == 'v1':
        return gpu_procs, gpu_procs_user
    elif version == 'v2':
        return gpu_procs, gpu_procs_user, gpu_procs_pid


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


def get_gpumem_utilization(allocated_gpus=[], username='all users', version='v1'):
    '''
    Gets the memory usage of each gpu
    '''
    if username != 'all users': pwd.getpwnam(username)

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
    mem_rates_user = [-1]*len(allocated_gpus)
    mem_rates_pid = [{}]*len(allocated_gpus)
    for i, gpuid in enumerate(allocated_gpus):
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        info = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetMemoryInfo, h,
                             ['something'])
        procs = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetComputeRunningProcesses, h,
                             ['something'])
        mem_rates[i] = int(np.ceil(100*info.used/info.total))
        mem_user = []
        mem_pid = {}
        for proc in procs:
            try:
                proc_uid = os.stat('/proc/%d' % (proc.pid)).st_uid
                if pwd.getpwuid(proc_uid).pw_name == username or username == 'all users':
                    mem_user.append(proc.usedGpuMemory)
            except:
                pass
            mem_pid[proc.pid] = int(np.ceil(100*proc.usedGpuMemory/info.total))
        mem_rates_user[i] = int(np.ceil(100*sum(mem_user)/info.total))
        mem_rates_pid[i] = mem_pid

    py3nvml.py3nvml.nvmlShutdown()

    if version == 'v1':
        return mem_rates
    elif version == 'v2':
        return mem_rates, mem_rates_user, mem_rates_pid


if __name__ == "__main__":
    stime = time.time()
    print(get_num_procs(version='v2', username='all users'))
    print(get_gpu_utilization())
    print(get_gpumem_utilization(version='v2', username='all users'))
    ftime = time.time()
    print(ftime - stime, 'seconds')