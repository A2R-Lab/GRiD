import os
import re
import sys
import platform
import subprocess
import pathlib
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                              ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.10.0':
                raise RuntimeError("CMake >= 3.10.0 is required on Windows")

        # Check for CUDA
        try:
            out = subprocess.check_output(['nvcc', '--version'])
            print("Found NVCC:", out.decode().strip())
        except OSError:
            raise RuntimeError("NVCC (CUDA Compiler) must be installed to build this extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPython_EXECUTABLE=' + sys.executable]

        # Specify CUDA architectures if not already in CMakeLists.txt
        # Uncomment and modify as needed for your target GPU
        # cmake_args += ['-DCMAKE_CUDA_ARCHITECTURES=75']  # For RTX 2000 series
        # cmake_args += ['-DCMAKE_CUDA_ARCHITECTURES=86']  # For RTX 3000 series
        # cmake_args += ['-DCMAKE_CUDA_ARCHITECTURES=89']  # For RTX 4000 series

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        
        # Set CUDA specific environment variables
        if 'CUDA_HOME' not in env and 'CUDA_PATH' in env:
            env['CUDA_HOME'] = env['CUDA_PATH']
        
        if 'CUDA_HOME' in env:
            print(f"Using CUDA from: {env['CUDA_HOME']}")
            if platform.system() == "Windows":
                env['PATH'] = f"{env['CUDA_HOME']}\\bin;{env['PATH']}"
            else:
                env['PATH'] = f"{env['CUDA_HOME']}/bin:{env['PATH']}"
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        print("Building with CMake arguments:", cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='gridCuda',
    version='0.1.0',
    author='Kwamena Awotwi',
    author_email='koawotwi@gmail.com',
    description='Python bindings for CUDA grid dynamics',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    ext_modules=[CMakeExtension('gridCuda')],
    cmdclass=dict(build_ext=CMakeBuild),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.6',
    zip_safe=False,
)
