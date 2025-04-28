from setuptools import find_packages, setup

package_name = 'autolap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mobilerobotics-008',
    maintainer_email='mobilerobotics-008@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ml = autolap.MLModel:main',
            'dash = autolap.RTC_Dash:main',
            'person_follower = loop.person_follower:main',
            'lane_follower = loop.lane_follower:main',
            'navigation_controller = loop.navigation_controller:main',
        ],
    },
)
