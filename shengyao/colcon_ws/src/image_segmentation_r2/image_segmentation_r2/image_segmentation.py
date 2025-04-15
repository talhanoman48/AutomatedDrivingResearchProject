#!/usr/bin/env python3

#
#  ==============================================================================
#  MIT License
#
#  Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright nothow to decleare parameters in ROS'
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ==============================================================================
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import os
from ament_index_python.packages import get_package_share_directory
from .img_utils import resize_image
import xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# rgb_to_class_id = {
#     (128, 64, 128):  0,   # Road
#     (244, 35, 232):  1,   # Sidewalk
#     (250, 170, 160): 2,   # Parking
#     (230, 150, 140): 3,   # Tail track
#     (220,  20,  60): 4,   # Person
#     (255,   0,   0): 5,   # Rider
#     (  0,   0, 142): 6,   # Car
#     (  0,   0,  70): 7,   # Truck
#     (  0,  60, 100): 8,   # Bus
#     (  0,  80, 100): 9,   # On Rails
#     (  0,   0, 230): 10,  # Motorcycle
#     (119,  11,  32): 11,  # Bicycle
#     (  0,   0,  90): 12,  # Caravan
#     (  0,   0, 110): 13,  # Trailer
#     ( 70,  70,  70): 14,  # Building
#     (102, 102, 156): 15,  # Wall
#     (190, 153, 153): 16,  # Fence
#     (180, 165, 180): 17,  # Guard Rail
#     (150, 100, 100): 18,  # Bridge
#     ( 50, 120,  90): 19,  # Tunnel
#     (153, 153, 153): 20,  # Pole
#     (220, 220,   0): 21,  # Traffic sign
#     (250, 170,  30): 22,  # Traffic light
#     (107, 142,  35): 23,  # Vegetation
#     (152, 251, 152): 24,  # Terrain
#     ( 70, 130, 180): 25,  # Sky
#     ( 81,   0,  81): 26,  # Ground
#     (111,  74,   0): 27,  # Dynamic
#     ( 20,  20,  20): 28,  # Static
#     (  0,   0,   0): 29   # None
# }

rgb_to_class_id = {
    (128, 64, 128):  0,   # Road
    (244, 35, 232):  1,   # Sidewalk
    (250, 170, 160): 2,   # Parking
    (230, 150, 140): 3,   # Tail track
    (220,  20,  60): 4,   # Person
    (255,   0,   0): 5,   # Rider
    (  0,   0, 142): 6,   # Car
    (  0,   0,  70): 7,   # Truck
    (  0,  60, 100): 8,   # Bus
    (  0,  80, 100): 9,   # On Rails
    (  0,   0, 230): 10,  # Motorcycle
    (119,  11,  32): 11,  # Bicycle
    (  0,   0,  90): 12,  # Caravan
    (  0,   0, 110): 13,  # Trailer
    ( 70,  70,  70): 14,  # Building
    (102, 102, 156): 15,  # Wall
    (190, 153, 153): 16,  # Fence
    (180, 165, 180): 17,  # Guard Rail
    (150, 100, 100): 18,  # Bridge
    ( 50, 120,  90): 19,  # Tunnel
    (153, 153, 153): 20,  # Pole
    (220, 220,   0): 21,  # Traffic sign
    (250, 170,  30): 22,  # Traffic light
    (107, 142,  35): 23,  # Vegetation
    (152, 251, 152): 24,  # Terrain
    ( 70, 130, 180): 25,  # Sky
    ( 81,   0,  81): 26,  # Ground
    (111,  74,   0): 27,  # Dynamic
    ( 20,  20,  20): 28,  # Static
    (157, 234,  50): 29,  # Road Line
    ( 45,  60, 150): 30,  # Water
    (  0,   0,   0): 31   # None
}

WITH_TF = True
try:
    import tensorflow as tf
except:
    WITH_TF = False
    print("%s will shutdown because it was not compiled with TensorFlow")

def parse_sample(image):
    """
    Argument:
    image_path -- String which contains the path to the camera image
    label_path -- String which contains the path to the label image
    
    Returns:
    image_rgb -- tf.Tensor of size [368, 1248, 3] containing the camera image
    label_segmentation_map -- tf.Tensor of size [368, 1248, 1] containing the segmentation map
    """
    ### START CODE HERE ### 
    #image_rgb = tf.image.decode_png(image, channels=3) 
    
    
    image_rgb = tf.image.resize(image, [512, 1024], method=tf.image.ResizeMethod.BILINEAR)  #can't be deleted, otherwise there would be Nonetype error
    
    # resize returns tf.float32 for BILINEAR, convert back to tf.uint8
    image_rgb = tf.cast(image, tf.uint8) 
          
    return image_rgb

def normalize(image):
    """
    Normalizes the input image from range [0, 255] to [0, 1.0]
    Arguments:
    image -- tf.tensor representing a RGB image with integer values in range [0, 255] 
    label -- tf.tensor representing the corresponding segmentation mask
    
    Returns:
    image -- tf.tensor representing a RGB image with integer values in range [0, 1] 
    label -- tf.tensor representing the corresponding segmentation mask
    """
    image = tf.cast(image, tf.float32) / 255.0  
    
    return image

class ImageSegmentation(Node):

    def predict(self, img_color_msg):
        t0 = time.time()

        # convert message to cv2-image
        input_img = self.cv_bridge.imgmsg_to_cv2(img_color_msg, desired_encoding="rgb8")

        # resize image
        input_img = resize_image(input_img, [self.resize_height, self.resize_width])
        
        # append batch dimension
        input_img = input_img[None]
        t1 = time.time()

        # perform semantic segmentation
        image = parse_sample(input_img)
        image = normalize(image)
        #image = tf.expand_dims(image, axis=0)
        probabilities= self.frozen_func(tf.cast(image, tf.float32))
        

        t2 = time.time()

        
        predictions = tf.argmax(probabilities, axis=-1)
        confidences=tf.reduce_max(probabilities,axis=-1)
        # remove batch dimension
        prediction = tf.squeeze(predictions).numpy()
        confidence=tf.squeeze(confidences).numpy()

        # decode image to RGB
        prediction = self.segmentation_map_to_rgb_encoding(prediction,rgb_to_class_id).astype(np.uint8)
        grayscale_confidence=confidence*255.0
        confidence_image = tf.cast(grayscale_confidence, tf.uint8).numpy()

        # convert output back to message
        seg_msg = self.cv_bridge.cv2_to_imgmsg(prediction, encoding="rgb8")
        confidence_msg=self.cv_bridge.cv2_to_imgmsg(confidence_image,encoding="mono8")

        # assign header of input msg
        seg_msg.header = img_color_msg.header
        confidence_msg.header=img_color_msg.header
        
        t3 = time.time()

        # log processing duration
        time_diffs = (t1 - t0, t2 - t1, t3 - t2, t3 - t0)
        log_message = "\n (prep) {:.2f}s | (pred) {:.2f}s | (post) {:.2f}s |(total) {:.2f}s".format(*time_diffs)
        self.get_logger().info(log_message)

        self.pub_seg.publish(seg_msg)
        self.pub_confidence.publish(confidence_msg)

    @staticmethod
    def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

    def load_frozen_graph(self, path_to_frozen_graph):
        self.sess = None
        self.graph = tf.Graph()

        self.input_tensor_name = 'x:0'
        self.output_tensor_name = 'Identity:0'

        with tf.io.gfile.GFile(path_to_frozen_graph, 'rb') as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(file_handle.read())
        
        # Wrap frozen graph to ConcreteFunctions
        self.frozen_func = self.wrap_frozen_graph(graph_def=graph_def,
                                                  inputs=[self.input_tensor_name],
                                                  outputs=[self.output_tensor_name],
                                                  print_graph=True)

    # def segmentation_map_to_rgb(self, segmentation_map):
    #     """
    #     Converts segmentation map to a RGB encoding according to self.color_palette
    #     Eg. 0 (Class 0) -> Pixel value [128, 64, 128] which is on index 0 of self.color_palette
    #         1 (Class 1) -> Pixel value [244, 35, 232] which is on index 1 of self.color_palette

    #     self.color_palette has shape [256, 3]. Each index of the first dimension is associated
    #     with an RGB value. The index corresponds to the class ID.

    #     :param segmentation_map: ndarray numpy with shape (height, width)
    #     :return: RGB encoding with shape (height, width, 3)
    #     """

    #     ### START CODE HERE ###
        
    #     # Task 1:
    #     # Replace the following command
    #     rgb_encoding = self.color_palette[segmentation_map]


    #     ### END CODE HERE ###

    #     return rgb_encoding
 
    def segmentation_map_to_rgb_encoding(self,segmentation_map, rgb_to_class_id):
        """
        Converts the segmentation map into a RGB encoding
        
        Arguments:
        segmentation_map -- Numpy ndArray of shape [height, width, 1]
        rgb_to_class_id -- Dictionary which contains the association between color and class ID
        
        Returns:
        rgb_encoding -- Numpy ndArray of shape [height, width, 3]
        """

        rgb_encoding = np.zeros([segmentation_map.shape[0], segmentation_map.shape[1], 3], dtype=np.uint8)
        
        ### START CODE HERE ### 
        for color, class_id in rgb_to_class_id.items():       # for color, class_id in None:
            
            rgb_encoding[segmentation_map==class_id] = color  # rgb_encoding[None==None] = None
        
        ### END CODE HERE ###
        return rgb_encoding

    
    def parse_convert_xml(self, conversion_file_path):
        """
        Parse XML conversion file and compute color_palette 
        """

        defRoot = ET.parse(conversion_file_path).getroot()

        color_to_label = {}

        color_palette = np.zeros((256, 3), dtype=np.uint8)
        class_list = np.ones((256), dtype=np.uint8) * 255
        class_names = np.array(["" for _ in range(256)], dtype='<U25')
        for idx, defElement in enumerate(defRoot.findall("SLabel")):
            from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
            to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
            class_name = defElement.get('Name').lower()
            if to_class in class_list:
                color_to_label[tuple(from_color)] = int(to_class)
            else:
                color_palette[idx] = from_color
                class_list[idx] = to_class
                class_names[idx] = class_name
                color_to_label[tuple(from_color)] = int(to_class)

        sort_indexes = np.argsort(class_list)

        class_list = class_list[sort_indexes]
        class_names = class_names[sort_indexes]
        color_palette = color_palette[sort_indexes]

        return color_palette, class_names, color_to_label

    def load_parameters(self):
        """
        Load ROS parameters and store them
        """
        self.get_logger().info("Loading parameters ...")

        # get the directory that this script is in
        package_dir = get_package_share_directory('image_segmentation_r2')
        
        # get the filename from the parameter and append it to the script directory
        frozen_graph_file = self.get_parameter('frozen_graph').get_parameter_value().string_value
        self.frozen_graph = os.path.join(package_dir,  frozen_graph_file)
        

        xml_conversion_file = self.get_parameter('xml_conversion_file').get_parameter_value().string_value
        self.path_xml_conversion_file = os.path.join(package_dir, xml_conversion_file)
        
        self.resize_width = self.get_parameter('resize_width').get_parameter_value().integer_value
        self.resize_height = self.get_parameter('resize_height').get_parameter_value().integer_value

        # self load one hot encoding
        self.color_palette, self.class_names, self.color_to_label = self.parse_convert_xml(self.path_xml_conversion_file)

    def setup(self):

        # create cv2-msg converter bridge
        self.cv_bridge = CvBridge()
        # load frozen graph
        self.load_frozen_graph(self.frozen_graph)
        # create publisher for passing on depth estimation and camera info      
        self.pub_seg = self.create_publisher(Image, "/image_segmented_shengyao", 0)
        self.pub_confidence=self.create_publisher(Image, "/image_confidence_shengyao", 0)
        # listen for input image and camera info
        self.sub_image = self.create_subscription(Image, "/image_color", self.predict, 0) # buff_size = 500 MB

    def __init__(self):
        # initialize ROS node
        super().__init__('camera_segmentation')

        # initialize ROS node
        self.get_logger().info("Initializing camera_segmentation node...")

        self.declare_parameter('frozen_graph', 'default_value')
        self.declare_parameter('xml_conversion_file', 'default_value')
        self.declare_parameter('resize_width', 1024)
        self.declare_parameter('resize_height', 512)

        if WITH_TF:

            # load parameters
            self.load_parameters()
 
            # setup components
            self.setup()

def main(args=None):
    rclpy.init(args=args)

    vision = ImageSegmentation()

    # keep node from exiting
    rclpy.spin(vision)
    
    #ROS2 needs .destroy_node after spinning
    vision.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
