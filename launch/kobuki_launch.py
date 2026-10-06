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
        'kobuki.yaml'
    )

    return LaunchDescription([
        # Ground Segmentation Node 
        Node(
            package=package_name,
            executable='ground_segmentation_node',
            name='seedbed_segmentation_node',
            output='screen',
            parameters=[system_config]  # Loading the unified YAML file
        ),        

        # Path Extraction Node 
        Node(
            package=package_name,
            executable='path_extraction_node',
            name='path_extraction_node',
            output='screen',
            parameters=[system_config]  # Loading the same unified YAML file
        ),
        
        # Camera Fusion Node 
        Node(
            package=package_name,
            executable='arducam_fusion_node', 
            name='arducam_fusion_node',
            output='screen',
            parameters=[ system_config]
        ),

        # Angular PID Node 
        Node(
            package=package_name,
            executable='kobuki_pid_node',       
            name='angular_pid_node',     
            output='screen',
            parameters=[system_config]   
        )
    ])