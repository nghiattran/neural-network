from distutils.core import setup
import mentality as package
import os.path


def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()

setup(
    name=package.__name__,
    packages=['lib'],
    version=package.__version__,
    py_modules=[package.__name__],
    description = package.__description__,
    author = package.__author__,
    author_email = package.__author_email__,
    url = package.__github__,
    long_description=read_file('README.md'),
    keywords = ['neural network', 'neural', 'network', 'perceptron'],
    classifiers = [],
)