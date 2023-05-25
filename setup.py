from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    description='A sample Python package',
    author='John Doe',
    author_email='jdoe@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
)