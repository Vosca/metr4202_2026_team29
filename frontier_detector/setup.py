from setuptools import find_packages, setup

package_name = 'frontier_detector'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='Thomas',
    maintainer_email='your_email@example.com',

    description=(
        'Frontier detection and clustering '
        'for TurtleBot3 exploration'
    ),

    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            (
                'frontier_detector = '
                'frontier_detector.frontier_detector:main'
            ),
        ],
    },
)
