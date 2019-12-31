#!/usr/bin/env python2
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage

class VisualizationFunction:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.dilate_number = 1.0 # dilate number for follow point
        self.black = np.array([0, 0, 0])  # class 0 (road/sky/etc.)
        self.yellow = np.array([1, 1, 0])  # class 1 (yellow line)
        self.white = np.array([1, 1, 1])  # class 2 (white line)
        self.green = np.array([0, 1, 0])  # class 3 (obstacles)
        self.red = np.array([1, 0, 0])  # class 4 (duckiebot)
        self.purple = np.array([1, 0, 1])  # class 5 (red line)
        self.blue = np.array([0, 0, 1])
        self.max_x_project = 0.8  # max x-position in ground projection (for visualization)
        self.max_y_project = 1.6  # max y-position in ground projection (for visualization)


    def visualize_ground_projection_and_follow(self,
                                        white_pos_x, white_pos_y,
                                        yellow_pos_x, yellow_pos_y,
                                        red_pos_x, red_pos_y,
                                        green_pos_x, green_pos_y,
                                        follow_x, follow_y):
        '''
        Function to create ground projection & follow point visualization (as an image).
        Input:
          - white_pos_x: list of white points x-positions in ground coordinate
          - white_pos_y: list of white points y-positions in ground coordinate
          - yellow_pos_x: list of yellow points x-positions in ground coordinate
          - yellow_pos_y: list of yellow points y-positions in ground coordinate
          - red_pos_x: list of red points x-positions in ground coordinate
          - red_pos_y: list of red points y-positions in ground coordinate
          - green_pos_x: list of green points x-positions in ground coordinate
          - green_pos_y: list of green points y-positions in ground coordinate
          - follow_point_x: follow point x-position in ground coordinate (float)
          - follow_point_y: follow point y-position in ground coordinate (float)
        Output:
          - msg_img: ground projection visualization images (CompressedImage)
        '''

        h_, w_ = 80, 160
        ground_project_im = np.zeros((h_, w_, 3))

        # For every point in the list, plot to image
        for i in range(len(white_pos_x)):
            x_im = h_ - (white_pos_x[i] / self.max_x_project) * h_  # convert (x) in robot frame to image coordinate frame
            y_im = w_ / 2 - (white_pos_y[i] / self.max_y_project) * (w_ / 2)  # convert (y) in robot frame to image coordinate frame
            x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
            ground_project_im[int(x_im), int(y_im)] = self.white
        for i in range(len(yellow_pos_x)):
            x_im = h_ - (yellow_pos_x[i] / self.max_x_project) * h_
            y_im = w_ / 2 - (yellow_pos_y[i] / self.max_y_project) * (w_ / 2)
            x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
            ground_project_im[int(x_im), int(y_im)] = self.yellow

        for i in range(len(red_pos_x)):
            x_im = h_ - (red_pos_x[i] / self.max_x_project) * h_
            y_im = w_ / 2 - (red_pos_y[i] / self.max_y_project) * (w_ / 2)
            x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
            ground_project_im[int(x_im), int(y_im)] = self.red

        for i in range(len(green_pos_x)):
            x_im = h_ - (green_pos_x[i] / self.max_x_project) * h_
            y_im = w_ / 2 - (green_pos_y[i] / self.max_y_project) * (w_ / 2)
            x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
            ground_project_im[int(x_im), int(y_im)] = self.green


        # Create 5 points to build a better visualization for follow point
        x_im = h_ - (follow_x / self.max_x_project) * h_
        y_im = w_ / 2 - (follow_y / self.max_y_project) * (w_ / 2)
        x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
        ground_project_im[int(x_im), int(y_im)] = self.blue

        x_im = h_ - (follow_x / self.max_x_project) * h_ + self.dilate_number
        y_im = w_ / 2 - (follow_y / self.max_y_project) * (w_ / 2) + self.dilate_number
        x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
        ground_project_im[int(x_im),int(y_im)] = self.blue

        x_im = h_ - (follow_x / self.max_x_project) * h_ - self.dilate_number
        y_im = w_ / 2 - (follow_y / self.max_y_project) * (w_ / 2) + self.dilate_number
        x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
        ground_project_im[int(x_im), int(y_im)] = self.blue

        x_im = h_ - (follow_x / self.max_x_project) * h_ - self.dilate_number
        y_im = w_ / 2 - (follow_y / self.max_y_project) * (w_ / 2) - self.dilate_number
        x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
        ground_project_im[int(x_im), int(y_im)] = self.blue

        x_im = h_ - (follow_x / self.max_x_project) * h_ + self.dilate_number
        y_im = w_ / 2 - (follow_y / self.max_y_project) * (w_ / 2) -self.dilate_number
        x_im, y_im = np.clip(x_im,0,79), np.clip(y_im,0,159)
        ground_project_im[int(x_im), int(y_im)] = self.blue

        ground_project_im_uint8 = (ground_project_im[:, :, ::-1] * 255).astype('uint8')  # This is BGR image
        ground_project_im_uint8 = cv2.dilate(ground_project_im_uint8, self.kernel) # dilate pixels to get a better visualization
        compressed_gp = np.array(cv2.imencode('.jpg', ground_project_im_uint8)[1]).tostring()
        msg_img = CompressedImage(header=None, format='jpeg', data=compressed_gp)

        return msg_img


    def visualize_ransac(self, white_points, yellow_points):
        '''
        This function is for making a ros_msg image with compressed format for ransac filter visualization
        Input:
          - white_points: white points position in pixel coordinate (list of tuple (int(x), float(y)))
          - yellow_points: yellow points position in pixel coordinate (list of tuple (int(x), float(y)))
        Output:
          - msg_img: ransac visualization image (CompressedImage)
        '''
        h_, w_ = 80, 160
        ransac_im = np.zeros((h_, w_, 3))

        # For every point in the list, plot to image
        for i in range(len(white_points)):
            x_im = white_points[i][0]
            y_im = white_points[i][1]
            ransac_im[int(x_im), int(y_im)] = self.white
        for i in range(len(yellow_points)):
            x_im = yellow_points[i][0]
            y_im = yellow_points[i][1]
            ransac_im[int(x_im), int(y_im)] = self.yellow

        # Convert to compressed image message
        ransac_im_uint8 = (ransac_im[:, :, ::-1] * 255).astype('uint8')  # This is BGR image
        ransac_im_uint8 = cv2.dilate(ransac_im_uint8, self.kernel) # dilate pixels to get a better visualization
        compressed_ransac_im = np.array(cv2.imencode('.jpg', ransac_im_uint8)[1]).tostring()
        msg_img = CompressedImage(header=None, format='jpeg', data=compressed_ransac_im)
        return msg_img
