from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import os
import cv2
from PIL import Image
from rosbags.image import message_to_cvimage
import numpy as np

image_count=0

output_dir = 'extracted_images'
os.makedirs(output_dir, exist_ok=True)

# Create a typestore.
typestore = get_typestore(Stores.LATEST)

# Create reader instance and open for reading.
with Reader('C:/Users/talha/Projects/carpp/rosbag') as reader:
    # Topic and msgtype information is available on .connections list.
    train_x = []
    train_y = []
    time1=0
    time2=1
    time1_prev=0
    time2_prev=1
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)
    # Iterate over messages.
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/carla/ego_vehicle/rgb/image':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            # get opencv image and convert to bgr8 color space
            img = message_to_cvimage(msg, "bgr8")
            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            time1 = msg.header.stamp.nanosec
        else:
            pass

        if connection.topic == '/carla/ego_vehicle/segmentation/image':
            msg_2 = typestore.deserialize_cdr(rawdata, connection.msgtype)
            # get opencv image and convert to bgr8 color space
            img_2 = message_to_cvimage(msg_2, "bgr8")
            im_rgb_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            time2 = msg_2.header.stamp.nanosec
        else:
             pass

        if time1==time2:
            if time1 != time1_prev and time2 != time2_prev:
                    Image.fromarray(im_rgb).save(f'extracted_images/train_x/image_{image_count:04d}.jpg')
                    Image.fromarray(im_rgb_2).save(f'extracted_images/train_y/image_{image_count:04d}.jpg')
                    image_count+=1
                    time1_prev = time1 
                    time2_prev = time2
            else:
                pass
        else:
            pass