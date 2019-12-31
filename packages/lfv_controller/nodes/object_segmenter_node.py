#!/usr/bin/env python2
#
#
# Description of 'object_segmenter' node:
# 1) This node subscribes to ROS image topic.
# 2) The ROS image is then converted to OpenCV image (ndarray), reversed (BGR -> RGB), and divided by 255.
# 3) The frame is then passed to semantic segmentation model which outputs the segmentation map
# 4) Publish the segmentation map
#========================================================================================

from __future__ import print_function

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from keras.models import Model, load_model
import numpy as np
import cv2
import os


class ObjectSegmenter:
    def __init__(self):
        self.vehicle = os.getenv("HOSTNAME")

        '''Load model'''
        if self.vehicle == 'default': # for simulation
            self.model = load_model('/duckietown/catkin_ws/src/lfv_core/packages/lfv_controller/nodes/sim_seg.h5')
        else: # for real world
            self.model = load_model('/workspace/custom_ws/src/lfv_controller/nodes/real_seg.h5')
        self.model.predict(np.ones((3, 160, 120, 3))) # always run the model once after loading the model
        self.height, self.width = 120, 160


        '''Initialize node and publishers'''
        rospy.init_node('object_segmenter')
        self.segmap_pub = rospy.Publisher('/object_segmenter_node/segmentation_map/compressed', CompressedImage, queue_size = 1) 
        

        '''Specify topic to subscribe from'''
        # self.image_sub = rospy.Subscriber('/{}/anti_instagram_node/corrected_image/compressed'.format(self.vehicle), CompressedImage, self.segment_callback, queue_size = 1)
        self.image_sub = rospy.Subscriber('/default/camera_node/image/compressed', CompressedImage, self.segment_callback, queue_size = 1, buff_size = 2**20)
        

        self.bridge = CvBridge()
        rospy.loginfo('Segmentation node initialized.')


    def get_path(self, filename):
        '''
        This function is for returning model abs path.
        Input:
          - filename: model's name (str)
        Output:
          - absolute path for the model (str)
        '''
        for r, ds, fs in os.walk("/"):
            for f in fs:
                fn = os.path.join(r, f)
                if f == filename:
                    return os.path.abspath(fn)


    def rosCompressedImageToArray(self, data):
        '''
        This function is to transform ROS CompressedImage into an numpy array
        Input:
          - data: RGB image (CompressedImage)
        Output:
          - frame: RGB image within [0,1] (ndarray)
        '''
        np_arr = np.fromstring(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_CUBIC)
        frame = frame[:, :, ::-1] / 255.0
        return frame


    def rosImageToArray(self, data):
        '''
        This function is to transform ROS Image message into an numpy array
        Input:
          - data: an RGB ROS image (Image)
        Output:
          - frame: an RGB image within [0,1] (ndarray)
        '''
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        frame = frame[:, :, ::-1] / 255.0
        return frame


    def color_segmap(self, data):
        '''
        This function is for transform class info within model into segmap(a framework of a segmentation map)
        Input:
          - data: data comes from model (ndarray)
        Output:
          - colored_segmap: a framework of a segmentation map(ndarray)
        '''
        black = np.array([0,0,0]) # class 0 (road/sky)
        yellow = np.array([1,1,0]) # class 1 (yellow line)
        white = np.array([1,1,1]) # class 2 (white line)
        green = np.array([0,1,0]) # class 3 (obstacles)
        red = np.array([1,0,0]) # class 4 (duckiebot)
        purple = np.array([1,0,1]) # class 5 (red line)
        cyan = np.array([0,1,1]) # class 6 (other objects)
        colored_segmap = np.zeros((self.height,self.width,3))
        for r in range(self.height):
            for c in range(self.width):
                if data[r,c] == 0:
                    colored_segmap[r,c,:] = black
                elif data[r,c] == 1:
                    colored_segmap[r,c,:] = yellow
                elif data[r,c] == 2:
                    colored_segmap[r,c,:] = white
                elif data[r,c] == 3:
                    colored_segmap[r,c,:] = green
                elif data[r,c] == 4:
                    colored_segmap[r,c,:] = red
                elif data[r,c] == 5:
                    colored_segmap[r,c,:] = purple
                elif data[r,c] == 6:
                    colored_segmap[r,c,:] = cyan
        return colored_segmap


    def segmap_to_image(self, data):
        '''
        This function is to transform a segmap into a BGR image
        Input:
          - data: a framework of a segmentation map (ndarray)
        Output:
          - colored_segmap_img_uint8: a BGR image (ndarray)
        '''
        segmap_img = np.argmax(data, axis=-1) # Size of (1, 160 * 120)
        segmap_img = np.reshape(segmap_img,(self.height,self.width)) # Size of (160, 120)
        colored_segmap_img = self.color_segmap(segmap_img)
        colored_segmap_img_uint8 = (colored_segmap_img[:,:,::-1] * 255).astype('uint8') # This is BGR image
        return colored_segmap_img_uint8


    def segment_callback(self, data):
        '''
        This function is for publishing segmentation maps with ros_msg Compressed format
        Input:
          - data: RGB image (CompressedImage)
        Output:
          - msg_img: segmentation map image (CompressedImage)
        '''

        '''Process the data to get segmentation map'''
        # frame = self.rosImageToArray(data) # If using Image message type as input
        frame = self.rosCompressedImageToArray(data) # If using CompressedImage message type as input
        frame = np.expand_dims(frame,0) # Input shape should be (1, 160, 120, 3) RGB
        segmap_flat = self.model.predict(frame) # segmap is an ndarray with shape (1, 160 * 120, 1)


        '''Construct message we want to publish'''
        segmap_uint8 = self.segmap_to_image(segmap_flat) # uint8 ndarray with shape (160, 120, 3) RGB
        compressed_segmap = np.array(cv2.imencode('.jpg', segmap_uint8)[1]).tostring()
        msg_img = CompressedImage(header = data.header, format = 'jpeg', data = compressed_segmap)
        try:
            self.segmap_pub.publish(msg_img)
        except CvBridgeError as e:
            rospy.logerr(str(e))


    def spin(self):
        rospy.spin()
    

if __name__ == '__main__':
    try:
        node = ObjectSegmenter()
        node.spin()
    except rospy.ROSInterruptException:
        pass
