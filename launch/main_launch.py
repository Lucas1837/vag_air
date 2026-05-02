import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'vitrox_project'

    # Path to your parameter file
    config1 = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'segmentation_params.yaml'
    )

    config2 = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'path.yaml'
    )

    return LaunchDescription([
        # 1. Ground Segmentation Node 
        Node(
            package=package_name,
            executable='ground_segmentation_node',
            name='ground_segmentation_node',
            output='screen',
            parameters=[config1]  # Loading the YAML file
        ),
        
        # 2. Path Extraction Node 
        Node(
            package=package_name,
            executable='path_extraction_node',
            name='path_extraction_node',
            output='screen',
            parameters=[config2]
        ),
        
        # 3. Camera Fusion Node 
        Node(
            package=package_name,
            executable='cam_fusion_node', # Ensure this matches how it's installed in CMakeLists
            name='cam_fusion_node',
            output='screen'
        )
    ])