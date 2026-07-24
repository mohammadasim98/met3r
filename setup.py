import os
from setuptools import setup
import subprocess
import sys
from distutils.cmd import Command
with open('requirements.txt') as f:
    required = f.read().splitlines()
__version__ = "1.1.0"


class GetSubmodules(Command):
    def run(self):
        subprocess.check_call(['git', 'submodule', 'update', "--init", "--recursive"])

setup(
        name="met3r", 
        version=__version__,
        author="Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, Jan Eric Lenssen",
        author_email="masim@mpi-inf.mpg.de, cwewer@mpi-inf.mpg.de, twimmer@mpi-inf.mpg.de, schiele@mpi-inf.mpg.de, jlenssen@mpi-inf.mpg.de",
        description="Official Code for 'MEt3R: Measuring Multi-View Consistency in Generated Images'",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/mohammadasim98/met3r',
        # Keep the upstream nested package layout: met3r/met3r.py resolves the
        # MASt3R and DUSt3R checkouts as siblings at runtime.
        packages=[
                "met3r",

                "mast3r/dust3r",
                "mast3r/dust3r/croco",
                "mast3r/dust3r/croco/models",
                "mast3r/dust3r/croco/models/curope",
                "mast3r/dust3r/croco/utils",
                "mast3r/dust3r/dust3r",
                "mast3r/dust3r/dust3r/cloud_opt",
                "mast3r/dust3r/dust3r/heads",
                "mast3r/dust3r/dust3r/utils",
                "mast3r/dust3r/dust3r/datasets",
                "mast3r/dust3r/dust3r/datasets/base",
                "mast3r/dust3r/dust3r/datasets/utils",

                "mast3r",
                "mast3r/mast3r",
                "mast3r/mast3r/cloud_opt",
                "mast3r/mast3r/cloud_opt/utils",
                "mast3r/mast3r/colmap",
                "mast3r/mast3r/utils",
                "mast3r/mast3r/datasets",
                "mast3r/mast3r/datasets/base",
                "mast3r/mast3r/datasets/utils",
        ],
        install_requires=[
                "torch",
                "torchvision",
                "iopath",
                "roma",
                "matplotlib",
                "tqdm",
                "opencv-python",
                "scipy",
                "einops",
                "numpy",
                "jaxtyping",
                "pytorch-lightning",
                "torchmetrics",
                "pyglet<2",
                "timm==0.4.12",
                "huggingface-hub[torch]>=0.22",
                "lpips",
                "featup@git+https://github.com/mhamilton723/FeatUp",
                "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git",
        ],
        extras_require={
                "known-camera": [
                        "pandas>=2.0",
                        "Pillow>=9.5",
                        "transformers>=4.56",
                        "natten",
                ],
        },
        entry_points={
                "console_scripts": [
                        "met3r-known-camera=met3r.cli:main",
                ],
        },

        classifiers=[
                'Programming Language :: Python :: 3',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
        ],
        python_requires='>=3.10',
        cmdclass={"submodule": GetSubmodules}
        
)
