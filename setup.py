from setuptools import setup, find_packages

setup(
    name='taming-transformers',
    version='0.0.1',
    description='Taming Transformers for High-Resolution Image Synthesis',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'tqdm',
        'regex',
        'ftfy',
        'boto3',  # BigGAN
        'requests'  # BigGAN
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
