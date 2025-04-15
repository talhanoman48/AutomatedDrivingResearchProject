import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from .img_utils import resize_image
import tensorflow as tf
from message_filters import ApproximateTimeSynchronizer, Subscriber

class SparseMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(SparseMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    #y_pred = tf.math.argmax(y_pred, axis=-1)#说明y_pred是独热编码
    return super().update_state(y_true, y_pred, sample_weight)
  


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
    (  0,   0,   0): 29   # None
}

def convert_rgb_encoding_to_segmentation_map(image, rgb_to_class_id):
    """
    Converts an image with the RGB class encoding into a class map.
    
    Argument:
    image -- tf.tensor of shape [heigh, width, 3] which contains for each pixel a tuple of (R, G, B) values.
    
    Returns:
    class_map -- tf.tensor of shape [heigh, width, 1] which contains for each pixel a single integer that represents a class
    """

    segmentation_map = tf.zeros([image.shape[0], image.shape[1]], dtype=tf.uint8)

    for color, class_id in rgb_to_class_id.items():    
    ### START CODE HERE ###
    
        segmentation_map = tf.where(
                                    condition=tf.reduce_all(tf.equal(image, color), axis=-1), # condition=None,
                                    x=tf.cast(class_id, tf.uint8),                            # x=None,
                                    y=segmentation_map                                        # y=None
                                    )
        
    # Add dimension to change the shape from [height, width] to [height, width, 1]
    segmentation_map = tf.expand_dims(segmentation_map, -1)  # segmentation_map = None
    ### END CODE HERE ###
        
    return segmentation_map

class Evaluation_Node(Node):

    def __init__(self):
        super().__init__('evaluation_node')
        self.subscription_pred = Subscriber(self,
            Image,
            '/image_segmented_shengyao')
            #self.prediction_callback,
            #0)
        self.subscription_label = Subscriber(self,
            Image,
            '/carla/ego_vehicle/semantic_front/image')
            #self.label_callback,
            #0)
        self.ts = ApproximateTimeSynchronizer(
            [self.subscription_pred, self.subscription_label], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.callback)
        self.publisher_ = self.create_publisher(Float32, 'miou', 0)
        self.bridge = CvBridge()
        self.predicted_image = None
        self.label_image = None
        self.metrics=SparseMeanIoU(num_classes=30,name='miou',dtype=tf.float32)
        self.syncFlag=0
        self.timestamp=None

    def callback(self, predicted_image, label_image):
        image1 = self.bridge.imgmsg_to_cv2(predicted_image, desired_encoding='rgb8')
        predicted=convert_rgb_encoding_to_segmentation_map(image1, rgb_to_class_id)
        predicted=tf.squeeze(predicted)
        predicted=tf.expand_dims(predicted,axis=0)
        self.predicted_image=predicted

        image2 = self.bridge.imgmsg_to_cv2(label_image, desired_encoding='rgb8')
        label = tf.image.resize(image2, [512, 1024], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 
        label_segmentation_map = convert_rgb_encoding_to_segmentation_map(label, rgb_to_class_id)  
        self.label_image=label_segmentation_map
        self.calculate_and_publish_miou()




    # def prediction_callback(self, msg):
    #     image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    #     predicted=convert_rgb_encoding_to_segmentation_map(image, rgb_to_class_id)
    #     predicted=tf.squeeze(predicted)
    #     predicted=tf.expand_dims(predicted,axis=0)
    #     self.predicted_image=predicted
    #     if msg.header.stamp.nanosec==self.timestamp.nanosec:
    #         self.calculate_and_publish_miou()
    #         self.syncFlag=1

        

    # def label_callback(self, msg):
    #     image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    #     label = tf.image.resize(image, [512, 1024], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 
    #     label_segmentation_map = convert_rgb_encoding_to_segmentation_map(label, rgb_to_class_id)  
    #     if self.label_image == None or self.syncFlag == 1:
    #         self.timestamp=msg.header.stamp
    #         self.label_image=label_segmentation_map
    #         self.syncFlag=0
    #     #self.calculate_and_publish_miou()

    def calculate_and_publish_miou(self):
        if self.predicted_image is not None and self.label_image is not None: 
            self.metrics.update_state(y_true=self.label_image,y_pred=self.predicted_image)
            miou = self.metrics.result()
            self.metrics.reset_state()# reset the state, otherwise the miou would be calulated accumulatedly
            msg = Float32()
            msg.data = float(miou)
            self.publisher_.publish(msg)

    #def calculate_miou(self, predicted, ground_truth, num_classes):
        # predicted_one_hot = tf.one_hot(predicted, depth=num_classes)
        # ground_truth_one_hot = tf.one_hot(ground_truth, depth=num_classes)

        # # Compute intersection and union
        # intersection = tf.reduce_sum(predicted_one_hot * ground_truth_one_hot, axis=[0, 1, 2])
        # union = tf.reduce_sum(predicted_one_hot + ground_truth_one_hot, axis=[0, 1, 2]) - intersection

        # # Compute IoU for each class
        # iou = intersection / (union + tf.keras.backend.epsilon())

        # # Compute mean IoU
        # miou = tf.reduce_mean(iou)
    #    return miou.numpy()

def main(args=None):
    rclpy.init(args=args)
    node = Evaluation_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
