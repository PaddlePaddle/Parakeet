# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import re
import six
import sys
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
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
        'numpy',
        'nltk',
        'inflect',
        'librosa',
        'unidecode',
        'numba',
        'tqdm',
        'matplotlib',
        'tensorboardX',
        'tensorboard',
        'scipy',
        'ruamel.yaml',
        'pandas',
        'sox',
        'soundfile',
        'llvmlite==0.31.0' if sys.version_info < (3, 6) else "llvmlite",
    ],

    # Package info
    packages=find_packages(exclude=('tests', 'tests.*')),
    zip_safe=True, )

setup(**setup_info)
