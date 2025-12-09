from setuptools import find_packages, setup

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        # torch/torchvision/opencv-python/pandas 등은 venv에서 pip로 설치하는 방식 권장
    ],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='ROS2 Jazzy YOLOv5n detection node with debug RGB overlay',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_inference_node = yolov5_ros.yolov5_node:main',
            'yolov5_distance_node = yolov5_ros.box_distance_node:main',
        ],
    },
)

