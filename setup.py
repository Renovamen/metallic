from os import path
from setuptools import setup, find_packages

current_path = path.abspath(path.dirname(__file__))

# get __version__ from `metallic/version.py`
__version__ = None
ver_file = path.join(current_path, 'metallic', 'version.py')
with open(ver_file) as fp:
    exec(fp.read())

# load content from `README.md`
def readme():
    readme_path = path.join(current_path, 'README.md')
    with open(readme_path, encoding = 'utf-8') as fp:
        return fp.read()

setup(
    name = 'metallic',
    version = __version__,
    packages = find_packages(),
    description = 'A clean, lightweight and modularized PyTorch meta-learning library.',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    keywords=['pytorch', 'meta-learning', 'few-shot learning'],
    license = 'MIT',
    author = 'Xiaohan Zou',
    author_email = 'renovamenzxh@gmail.com',
    url = 'https://github.com/Renovamen/metallic',
    include_package_data = True,
    install_requires = [
        'numpy>=1.14.0,<1.20.0',
        'torch>=1.4.0',
        'torchvision>=0.5.0',
        'higher>=0.2.1',
        'requests',
        'tqdm',
    ]
)
