from distutils.core import setup

setup(
    name='airush-2022-pubtrans',
    version='1.0',
    install_requires=[
        'pandas',
        'pyarrow',
        'numpy',
        'fastparquet',
        'torch == 1.8.0',
        'tqdm',
        'scikit-learn',
        
    ],
    python_requires='>=3.6',
)
