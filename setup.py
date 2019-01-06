""" Setup """

import re
from setuptools import setup, find_packages

with open('README.md', 'rb') as f:
    DESCRIPTION = f.read().decode('utf-8')

with open('timeflux_dsp/__init__.py') as f:
    VERSION = re.search('^__version__\s*=\s*\'(.*)\'', f.read(), re.M).group(1)

setup(
    name='Timeflux dsp plugin',
    packages=find_packages(),
    version=VERSION,
    description='A Timeflux plugin for signal processing.',
    long_description=DESCRIPTION,
    author='Raphaelle Bertrand-Lalo',
    author_email='contact@timeflux.io',
    url='https://timeflux.io',
)
