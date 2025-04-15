import tensorflow as tf
import random
import os
import re
import streamlit as st

random.seed(1234)

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
        segmentation_map = tf.where(
                                    condition=tf.reduce_all(tf.equal(image, color), axis=-1), # condition=None,
                                    x=tf.cast(class_id, tf.uint8),                            # x=None,
                                    y=segmentation_map                                        # y=None
                                    )
    # Add dimension to change the shape from [height, width] to [height, width, 1]
    segmentation_map = tf.expand_dims(segmentation_map, -1)  # segmentation_map = None
        
    return segmentation_map

def parse_sample(image_path, label_path):
    """
    Argument:
    image_path -- String which contains the path to the camera image
    label_path -- String which contains the path to the label image
    
    Returns:
    image_rgb -- tf.Tensor of size [368, 1248, 3] containing the camera image
    label_segmentation_map -- tf.Tensor of size [368, 1248, 1] containing the segmentation map
    """
    ### START CODE HERE ### 
    image_rgb = tf.image.decode_png(tf.io.read_file(image_path), channels=3) 
    label_rgb = tf.image.decode_png(tf.io.read_file(label_path), channels=3)  
    
    image_rgb = tf.image.resize(image_rgb, [512, 1024], method=tf.image.ResizeMethod.BILINEAR)  #can't be deleted, otherwise there would be Nonetype error
    label_rgb = tf.image.resize(label_rgb, [512, 1024], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  #
    # resize returns tf.float32 for BILINEAR, convert back to tf.uint8
    image_rgb = tf.cast(image_rgb, tf.uint8) 
    
    # apply convert_rgb_encoding_to_segmentation_map to the label_rgb image
    label_segmentation_map = convert_rgb_encoding_to_segmentation_map(label_rgb, rgb_to_class_id)  
    ### END CODE HERE ###
    
    return image_rgb, label_segmentation_map

def getPaths(folder_path='C:/Users/talha/Projects/carpp/combined/'):
    folder_path=folder_path
    images_path=[]
    labels_path=[]
    pattern='.*?/x'

    for foldername, subfolders, filenames in os.walk(folder_path):

        if re.match(pattern,foldername)!=None:
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                images_path.append(file_path)

    pattern='.*?/y'
    for foldername, subfolders, filenames in os.walk(folder_path):
        if re.match(pattern,foldername)!=None:
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                labels_path.append(file_path)
    # shuffle the dataset
    fused_list = list(zip(images_path, labels_path))

    images_path_train, labels_path_train = map(list, zip(*fused_list))

    return images_path_train, labels_path_train

def normalize(image, label):
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
    
    return image, label

def create_dataset(images_path, labels_path, batch_size=128, buffer_size=500, do_augmentation=False):
    """
    Dataset creation function. Creates a input pipeline for semantic image segmentation.
    
    Arguments:
    - images_path -- List of Strings which contain pathes for the camera images
    - labels_path -- List of Strings which contain pathes for the label images
    - batch_size -- Integer - Size of the batches during data creation
    - buffer_size -- Integer - Size of the buffer for shuffeling
    - do_augmentation -- Boolean - If True, apply data augmentation
    
    Returns:
    - dataset -- tf.data.Dataset
    """
    
    dataset = tf.data.Dataset.from_tensor_slices((images_path, labels_path))

    # Apply the parse_sample function. Use tf.data.AUTOTUNE for the number of parallel calls
    dataset = dataset.map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    # Apply batching to the dataset using batch_size
    dataset = dataset.batch(batch_size=batch_size)
    # Use prefetching 
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    return dataset

