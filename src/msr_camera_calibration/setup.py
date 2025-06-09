import os
from glob import glob
from setuptools import find_packages, setup


package_name = 'msr_camera_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         *[(os.path.join('share', package_name, os.path.dirname(f)), [f]) 
        for f in glob('data/**/*', recursive=True) if os.path.isfile(f)]
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
            'get_homography = msr_camera_calibration.get_homography:main',
            'checkerboard_image_saver = msr_camera_calibration.checkerboard_image_saver:main',
            'camera_calibration = msr_camera_calibration.camera_calibration:main',
            'undistort_frames = msr_camera_calibration.undistort_frames:main',
            'calib_img_thinner = msr_camera_calibration.calib_img_thinner:main',
        ],
    },
)
