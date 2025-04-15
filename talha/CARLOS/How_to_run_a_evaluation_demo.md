## How to run a evaluation demo 

#### Modify the town


1. change the line 49 of the file `/carlos/software-prototypingcarla-ros-bridge.launch.py` in carlos repository.
2. change the default_value at line 49 from "Town10HD" to "Town07". Notice the "Town07" is not the standard map, so the container should be edited according to `./How to add the extra maps into carlos.md`
3. Then go back to the `/carlos` and run the `./run_demo.sh` again
4. go to directory `./docker-clion` and run the command `docker compose up`
5. At the very beginning the model will fall out of the world, after a minute or two it can be automatically recovered


#### Modify the camera setting

1. edit the camera image_x and image_y in `/carlos/software-prototyping/sensors.json`


#### Modify the models

1. edit the file `/shengyao/colcon_ws/src/image_segmentation_r2/config/params.yaml`