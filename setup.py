
import os
from glob import glob
from setuptools import setup

package_name = 'uuv_fresh'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joel',
    maintainer_email='joel@todo.todo',
    description='Fresh URDF + keyboard drive in RViz',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
         'console_scripts': [
           'rrtstar_nav = uuv_fresh.rrtstar_nav:main',
           'cmdvel_6dof_to_tf = uuv_fresh.cmdvel_6dof_to_tf:main',
           'teleop_3d_keyboard = uuv_fresh.teleop_3d_keyboard:main',
           'dose_cloud_combined_csv_pub = uuv_fresh.dose_cloud_combined_csv_pub:main',
           'dose_cloud_to_voxel_box_split = uuv_fresh.dose_cloud_to_voxel_box_split:main',
           'rrtstar_nav_6dof = uuv_fresh.rrtstar_nav_6dof:main',
         ],
    },
)
