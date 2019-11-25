import os 
import io 
import re
from setuptools import setup, find_packages

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = find_version('parakeet', '__init__.py')
long_description = read('README.md')

setup_info = dict(
    # Metadata
    name='parakeet',
    version=VERSION,
    author='PaddleSL Team',
    author_email='',
    url='https://github.com/PaddlePaddle',
    description='Speech synthesis tools and models based on Paddlepaddle',
    long_description=long_description,
    license='Apache 2',

    install_requires=[
        'numpy', 'nltk', 'inflect', 'librosa', 'unidecode', 'numba', 
        'tqdm', 'matplotlib', 'tensorboardX', 'tensorboard', 'scipy',
        'ruamel.yaml', 'pandas', 'sox',  
    ],

    # Package info
    packages=find_packages(exclude=('tests', 'tests.*')),

    zip_safe=True,
)

setup(**setup_info)