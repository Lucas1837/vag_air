import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'vitrox_project'

    # Single path to the unified parameter file
    system_config = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'system_params.yaml'
    )

    return LaunchDescription([
        # 1. Ground Segmentation Node 
        Node(
            package=package_name,
            executable='ground_segmentation_node',
            name='seedbed_segmentation_node',
            output='screen',
            parameters=[system_config]  # Loading the unified YAML file
        ),
        
        # Node(
        #     package=package_name,
        #     executable='pcl_segmentation_node',
        #     name='pcl_segmentation_node',
        #     output='screen',
        #     parameters=[system_config]  # Loading the unified YAML file
        # ),

        # 2. Path Extraction Node 
        Node(
            package=package_name,
            executable='path_extraction_node',
            name='path_extraction_node',
            output='screen',
            parameters=[system_config]  # Loading the same unified YAML file
        ),
        
        # 3. Camera Fusion Node 
        Node(
            package=package_name,
            executable='cam_fusion_node', 
            name='cam_fusion_node',
            output='screen'
        ),

        # 4. Angular PID Node (Newly Added)
        # Node(
        #     package=package_name,
        #     executable='pid_node',       # Matches the executable name in CMakeLists.txt
        #     name='angular_pid_node',     # Matches the Node("...") name inside pid.cpp
        #     output='screen',
        #     parameters=[system_config]   # Loading the same unified YAML file
        # )
    ])