import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'vitrox_project'

    urdf_file_path = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        'robot.urdf' 
    )

    with open(urdf_file_path, 'r') as infp:
        robot_desc = infp.read()

    # ==========================================
    # THE MAGIC FIX: Convert package:// to an absolute file:// path
    # ==========================================
    pkg_share_dir = get_package_share_directory(pkg_name)
    robot_desc = robot_desc.replace('package://vitrox_project', 'file://' + pkg_share_dir)

    # ==========================================
    # 1. Start Gazebo Fortress
    # ==========================================
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # ==========================================
    # 2. Start the Robot State Publisher
    # ==========================================
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_desc}]
    )

    # ==========================================
    # 3. Spawn the Robot into Gazebo
    # ==========================================
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-string', robot_desc,
            '-name', 'ranger_robot',
            '-allow_renaming', 'true',
            '-z', '0.5' 
        ]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])