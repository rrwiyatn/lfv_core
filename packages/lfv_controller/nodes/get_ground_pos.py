#! /usr/bin/env python
# This is modified version of the original ground_projection package (https://github.com/duckietown/dt-core/tree/daffy/packages/ground_projection)
import numpy as np
from duckietown_msgs.msg import Pixel
from geometry_msgs.msg import Point
# from ground_projection.srv import GetGroundCoord
from duckietown_utils import get_duckiefleet_root
from duckietown_utils.yaml_wrap import yaml_load_file
import rospy
import os



class Pixel2Ground:
    def __init__(self):
        self.vehicle = os.getenv("HOSTNAME") # get duckiebot name
        self.height = 80
        self.width = 160
        self.noise_range = 0.03 # (-0.03, 0.03), noise range
        self.H = self.load_homography(self.vehicle) # 3 x 3 homography matrix


    def load_homography(self, robotname):
        '''
        Function to load homography file for ground projection.
        Input:
          - robotname: e.g., "default" if using simulator
        Output:
          - homo_mat: homography matrix (3 x 3 ndarray)
        '''
        if robotname == "default":
            homo_path = (get_duckiefleet_root() + "/calibrations/camera_extrinsic/default.yaml") # simulator
        else:
            homo_path = (get_duckiefleet_root() + "/calibrations/camera_extrinsic/"+ robotname +".yaml") # real duckiebot
        data = yaml_load_file(homo_path)
        homo_mat = np.array(data['homography']).reshape((3, 3))
        # homo_mat =  np.array([-1.27373e-05, -0.0002421778, -0.1970125, 0.001029818, -1.578045e-05, -0.337324, -0.0001088811, -0.007584862, 1.]).reshape((3, 3))
        return homo_mat
        

    def pixel2ground_local(self, points):
        '''
        Function to apply ground projection.
        Input:
          - points: list of points position in pixel coordinate (list of tuple (int(x), int(y)))
        Output:
          - x_list: list of x-position(float) of input points in ground coordinate for input points
          - y_list: list of y-position(float) of input points in ground coordinate for input points
        '''
        cw = 640.0
        ch = 480.0
        x_list = []
        y_list = []
        for i in range(len(points)):
            if int(points[i][0]) >=30:
                v = int((float(points[i][0]) / float(self.height)) * ch)
                u = int((float(points[i][1]) / float(self.width)) * cw)
                uv_raw = np.array([u, v])
                uv_raw = np.append(uv_raw, np.array([1]))
                ground_point = np.dot(self.H, uv_raw)
                x = ground_point[0]
                y = ground_point[1]
                z = ground_point[2]
                x_list.append(x / z)
                y_list.append(y / z)
        return x_list, y_list


    def add_gaussian_noise(self, x, y):
        '''
        Function is to add Gaussian noise to ground projected points
        Input:
          - x: list of points x-position(float) in ground coordinate
          - y: list of points y-position(float) in ground coordinate
        Output:
          - new_x: list of new points x-position(float) in ground coordinate
          - new_y: list of new points y-position(float) in ground coordinate
        '''
        new_x = []
        new_y = []
        for i in range(len(x)):
            noise = np.random.random() * 2. - 1.0
            noise = noise * self.noise_range
            x_ = x[i] + noise
            y_ = y[i] + noise
            new_x.append(x_)
            new_y.append(y_)
        return new_x, new_y
