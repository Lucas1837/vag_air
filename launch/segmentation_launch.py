import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Path to your YAML file
    config = os.path.join(
        get_package_share_directory('vitrox_project'), # <-- FIXED THIS
        'config',
        'segmentation_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='vitrox_project',               # <-- FIXED THIS
            executable='ground_segmentation_node',  # <-- MAKE SURE THIS MATCHES YOUR CMAKELISTS EXECUTABLE NAME
            name='ground_segmentation_node', 
            output='screen',
            parameters=[config]              
        )
    ])