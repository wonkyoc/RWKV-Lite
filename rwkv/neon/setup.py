from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import subprocess

mcpu = None
march = None
flag = ''

# Run 'lscpu' command to check for the CPU model
cpu_info = subprocess.check_output("lscpu", shell=True).decode()
if "A53" in cpu_info:   # opi0
    mcpu = "cortex-a53"
    march = "armv8-a"
elif "A76" in cpu_info: # rpi5
    mcpu = "cortex-a76"
    march = "armv8.2-a+fp16"
    flag = "-DHAS_NEON_FP16"
elif "A72" in cpu_info:  # rpi4
    mcpu = "cortex-a72"
    march = "armv8-a"
else: 
    print(f"Could not detect CPU model: exit")
    sys.exit(1)
    
setup(
    name='mm8_neon',
    ext_modules=[
        CppExtension(
            'mm8_neon',
            ['mm8_neon.cpp'],
            extra_compile_args=[
                '-O3',
                '-g',
                f'-mcpu={mcpu}',
                f'-march={march}',
                '-std=c++17',
                flag,
                '-fopenmp'    # Enable OpenMP
            ],
            extra_link_args=['-fopenmp'],  # Link against OpenMP library
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
