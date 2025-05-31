from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'msr_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [
            'resource/' + package_name
        ]),
        ('share/' + package_name, ['package.xml']),
        # Include everything in launch, urdf, meshes, worlds, etc.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'run'), glob('run/*')),
        (os.path.join('share', package_name, 'models/puzzlebot'), glob('models/puzzlebot/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roger',
    maintainer_email='joserogelioruiz12@gmail.com',
    description='Puzzlebot Gazebo and RViz simulation setup',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_tf_broadcaster = msr_simulation.odom_tf_broadcaster:main',
            'simulation_odometry = msr_simulation.simulation_odometry:main',
            'trajectory_commander = msr_simulation.trajectory_commander:main',
            'pose_saver = msr_simulation.pose_saver:main', 
        ],
    },

)
