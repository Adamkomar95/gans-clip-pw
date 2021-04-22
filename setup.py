from setuptools import setup, find_packages

setup(
    name='pw-gans-clip',
    packages=find_packages(),
    include_package_data=True,
    version='0.0.0',
    license='MIT',
    description='PW-GANS-CLIP',
    author='Adam Komorowski, Maciej Domagała',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'text to image',
        'generative adversarial networks'
    ],
    install_requires=[
        'numpy'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)