from setuptools import find_packages, setup

package_name = 'msr_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[  
        'setuptools',
        'numpy',
        'opencv-python',
        'cv_bridge',
        ],
    zip_safe=True,
    maintainer='idmx',
    maintainer_email='a00227388@tec.mx',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_map_event = msr_detection.yolo_map_event:main',
        ],
    },
)
