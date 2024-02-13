from setuptools import setup, find_packages

setup(
    name='UTME',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'networkx',
        'sentence-transformers',
        'openai==0.28',
        'tqdm',
    ],
)
