from os import path
from setuptools import setup, find_packages

current_path = path.abspath(path.dirname(__file__))

# get __version__ from `metacraft/version.py`
__version__ = None
ver_file = path.join(current_path, 'metacraft', 'version.py')
with open(ver_file) as fp:
    exec(fp.read())

# load content from `README.md`
def readme():
    readme_path = path.join(current_path, 'README.md')
    with open(readme_path, encoding = 'utf-8') as fp:
        return fp.read()

setup(
    name = 'metacraft',
    version = __version__,
    packages = find_packages(),
    description = 'A simple toolbox for meta-learning research based on Pytorch.',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    author = 'Xiaohan Zou',
    author_email = 'renovamenzxh@gmail.com',
    url = 'https://github.com/Renovamen/metacraft',
    include_package_data = True
)