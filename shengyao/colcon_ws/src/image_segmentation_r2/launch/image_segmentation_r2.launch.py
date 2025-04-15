import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():

    # Get the package and params directory
    image_segmentation_dir = get_package_share_directory('image_segmentation_r2')
    config = os.path.join(image_segmentation_dir, "config","params.yaml")

    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock time')


        # CAMERA SEGMENTATION NODE
    camera_segmentation_node = Node(
        package='image_segmentation_r2',
        name='image_segmentation_r2',
        executable='image_segmentation',
        output='screen',
        parameters=[config],
        remappings=[
            ('image_color', '/carla/ego_vehicle/rgb_front/image')
        ]
    )
    evaluation_node = Node(
        package='evaluation_node',
        name='evaluation_node',
        executable='evaluation_node',
        output='screen',
        
    )

       

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add the actions to the launch description
    ld.add_action(use_sim_time)  
    ld.add_action(camera_segmentation_node)
    ld.add_action(evaluation_node)
    return ld
