import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'line_follow_msr'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob(os.path.join('data', '*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roger',
    maintainer_email='joserogelioruiz12@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_follower_samu = line_follow_msr.line_follower_samu:main',
            'line_follower_gordo = line_follow_msr.line_follower_gordo:main',
            'line_follower_hamburguesota = line_follow_msr.line_follower_hamburguesota:main',
            'line_follower_obesidad = line_follow_msr.line_follower_obesidad:main',
            'line_follower_sobrepeso = line_follow_msr.line_follower_sobrepeso:main',
            'line_follower_bigChungus = line_follow_msr.line_follower_bigChungus:main',
            'line_follower_bigbigChungus = line_follow_msr.line_follower_bigbigChungus:main',
            'line_follower_obesidad_I = line_follow_msr.line_follower_obesidad_I:main',
            'line_follower_obesidad_I_for_launch = line_follow_msr.line_follower_obesidad_I_for_launch:main',
            'line_follower_THICK = line_follow_msr.line_follower_THICK:main',
            'line_follower_WesternBacon = line_follow_msr.line_follower_WesternBacon:main',
            'crossroad_detection = line_follow_msr.crossroad_detection:main',
            'crossroad_detection_for_launch = line_follow_msr.crossroad_detection_for_launch:main',
            'line_follower_obesidad_II = line_follow_msr.line_follower_obesidad_II:main',
            'turn_manager_node = line_follow_msr.turn_manager_node:main',
        ],
    },
)
