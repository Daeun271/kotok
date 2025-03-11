from setuptools import setup

def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='kotok',
    version='1.0.0',
    packages=['kotok'],
    install_requires=read_requirements(),
    author='Daeun Jung',
    author_email='Daeun.Jung@ruhr-uni-bochum.de',
    description='Korean morphological analyzer based on the BERT architecture',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/Daeun271/kotok',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
