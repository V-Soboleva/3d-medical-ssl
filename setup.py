from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='pixelwise-ssl',
    version='0.0.1',
    description='3d Self Supervised Learning for Medical Images.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/V-Soboleva/3d-medical-ssl',
    packages=find_packages(include=('3d_madical_ssl',)),
    python_requires='>=3.6',
    install_requires=requirements,
)