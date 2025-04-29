from setuptools import setup

package_name = 'autolap'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.loop'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/autolap_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Autonomy stack for following lanes and people',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_follower = autolap.loop.person_follower:main',
            'lane_follower = autolap.loop.lane_follower:main',
            'navigation_controller = autolap.loop.navigation_controller:main',
        ],
    },
)
