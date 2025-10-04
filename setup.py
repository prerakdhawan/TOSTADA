from setuptools import setup, find_packages

setup(
    name='tostada',
    version='0.3.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'tostada': ['util/*.csv', 'util/*.txt'],
    },
    description='Toolkit for spatially tailored disordered arrangements',
    author='Prerak Dhawan',
)