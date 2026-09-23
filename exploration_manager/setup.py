from glob import glob

from setuptools import find_packages, setup


package_name = 'exploration_manager'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [
                'resource/' + package_name
            ]
        ),
        (
            'share/' + package_name,
            [
                'package.xml'
            ]
        ),
        (
            'share/' + package_name + '/launch',
            glob(
                'launch/*.launch.py'
            )
        ),
        (
            'share/' + package_name + '/config',
            glob(
                'config/*.yaml'
            )
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='Thomas',
    maintainer_email='your_email@example.com',

    description=(
        'Frontier-based exploration manager '
        'for TurtleBot3'
    ),

    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            (
                'exploration_manager = '
                'exploration_manager.'
                'exploration_manager:main'
            ),
        ],
    },
)
