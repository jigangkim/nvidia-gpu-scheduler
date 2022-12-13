import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvidia_gpu_scheduler",
    version="1.1.2",
    author="Jigang Kim",
    author_email="jgkim2020@snu.ac.kr",
    description="NVIDIA GPU compute task scheduling utility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jigangkim/nvidia-gpu-scheduler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        'ipython',
        'numpy',
        'py3nvml',
        'python-dateutil',
        'tqdm',
        'gin-config'
    ],
)