#!/bin/bash
/etc/init.d/ssh restart
source /opt/ros/humble/setup.bash #注意需要source原装setup.bash
cd /docker-ros/ws
colcon build
chmod -R 777 /docker-ros/ws
source /docker-ros/ws/install/setup.bash
ros2 launch image_segmentation_r2 image_segmentation_r2.launch.py
#top -b #a foreground process needed, otherwise the container would automatically exit
