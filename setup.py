""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('gen_efficientnet/version.py').read())
setup(
    name='gen_efficientnet',
    version=__version__,
    description='PyTorch (Generic) EfficientNets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rwightman/gen-efficientnet-pytorch',
    author='Ross Wightman',
    author_email='hello@rwightman.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained models efficientnet mixnet mobilenetv3 mnasnet',
    packages=find_packages(exclude=['convert']),
    install_requires=['torch >= 1.1', 'torchvision'],
    python_requires='>=3.6',
)
