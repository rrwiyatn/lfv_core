#! /usr/bin/env python
import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from cut_line_filter import ImgPixelsProcess
from get_ground_pos import Pixel2Ground
from duckietown_msgs.msg import Pixel
from visualization import VisualizationFunction
from dt_project_utils import SpaceInfo, AntiBadPrediction
import os
import time


# Description of 'lfv_controller' node:
# 1) This node subscribes to the segmentation map published by "object_segmenter_node".
# 2) Segmentation map is then converted to OpenCV image (ndarray), reversed (BGR -> RGB), and divided by 255.
# 3) Crop image to 80 x 160
# 4) Cropped image is then processed to return 4 classes points' position in pixel coordinate
# 5) These points are then filtered
# 6) Fit white and yellow lines with RANSAC
# 7) Ground project these points
# 8) Compute follow point
# 9) Compute v and omega, then publish this command to be executed


class LFVController:
    def __init__(self):
        self.img_process = ImgPixelsProcess() # init image processing related class
        self.p2g = Pixel2Ground() # init ground projection related class
        self.visual = VisualizationFunction() # init visualization related class
        self.spaceinfo = SpaceInfo() # init space info compute related class
        self.anti_noise = 30 # if the number of one class of points is less than this number, will be regarded as noise
        
        
        ##################################################
        ########### EXPERIMENTAL PARAMETERS ##############
        ########## NOT NEEDED FOR LFV CHALLENGE ##########
        # self.anti_bad_prediction = AntiBadPrediction() # init anti bad prediction related class (EXPERIMENTAL)
        self.assumed_lane_width = 1.0
        self.avoid_obstacle = False # if True, will enable obstacle avoidance if there is enough opening (EXPERIMENTAL)
        if self.avoid_obstacle:
            self.assumed_lane_width = 1.0
            self.bias_collision = 0.5  # offset value for collision avoidance
        self.safe_dis_green = 0.33 # safe distance to green objects
        self.safe_dis_red = 0.5 # safe distance to duckiebot
        self.green_stop = False # if True, will consider duckies/cones/barricades as obstacles. If False, only consider Duckiebots as obstacles.
        ##################################################
        ##################################################


        ##################################################
        ########### ADJUSTABLE PARAMETERS ################
        self.enable_visualization = True  # if True, output of ransac and ground projection will be visualized as images. Set to False if not needed so we can speed things up.
        self.v = 0.3  # velocity multiplier used in getSpeed()
        self.bias = 0.7  # offset value for follow point
        self.weight_white = 0.55  # When seeing both white and yellow lines, one may choose to give more weights to white or yellow lines to compute follow point
        self.weight_yellow = 1. - self.weight_white
        self.v_max = 0.7  # max possible speed
        self.v_min = 0.0  # min possible speed
        self.k_omega_both = 1.5 # omega gain when seeing both white and yellow lines, the bigger it is, the slower duckiebot will turn
        self.k_omega_yellow = 0.4 # omega gain when only seeing yellow line, the bigger it is, the slower duckiebot will turn
        self.k_omega_white = 0.4 # omega gain when only seeing white line, the bigger it is, the slower duckiebot will turn
        self.v_factor = 1.0 # speed adjustment gain, big number allows the robot to accelerate and decelerate faster during straight lane and hard turns, respectively
        self.position_ground = 0.4 # closest distance allowed between robot and obstacles (i.e., distance before stopping)
        ##################################################
        ##################################################


        rospy.init_node('lfv_controller', anonymous=False)

        # Subscriber
        self.sub = rospy.Subscriber("/object_segmenter_node/segmentation_map/compressed", CompressedImage, self.callback, queue_size = 1, buff_size = 2**20) # subscriber to segmentation map

        # Publisher
        self.pub = rospy.Publisher('~car_cmd', Twist2DStamped, queue_size=1) # publisher for v and omega
        self.viz_ransac_pub = rospy.Publisher('/ransac/image/compressed', CompressedImage, queue_size = 1) # To visualize RANSAC output
        self.viz_ground_project_pub = rospy.Publisher('/ground_project/image/compressed', CompressedImage, queue_size = 1) # To visualize ground projection output

        rospy.loginfo('LFV controller node initialized.')


    def rosCompressedImageToArray(self, data):
        '''
        This function is to transform ROS CompressedImage into an numpy array.
        Input:
          - data: RGB image (CompressedImage)
        Output:
          - frame: RGB image within [0,1] (ndarray)
        '''
        np_arr = np.fromstring(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = frame[:, :, ::-1] / 255.0 # RGB image
        return frame


    def getSpeed(self, follow_point_x, follow_point_y, k_omega):
        '''
        Function to compute v and omega.
        Input:
          - follow_point_x: follow point's x position in ground coordinate (float)
          - follow_point_y: follow point's y position in ground coordinate (float)
        Output:
          - v: line velocity for duckiebot (float)
          - omega: angle velocity for duckiebot (float)
        '''
        real_d = math.sqrt(follow_point_x ** 2 + follow_point_y ** 2)
        sinalpha = follow_point_y / real_d
        v = (self.v) / ((abs(sinalpha + 1e-5) * 2.0) ** self.v_factor) # adjust velocity based on alpha
        omega = sinalpha / k_omega
        v = np.clip(v, self.v_min, self.v_max) # limit v within v_min and v_max
        # print('velocity: %.3f, omega:%.3f' % (v, omega))
        return v, omega


    def point_filter(self, white_points, yellow_points, red_points, green_points):
        '''
        This function is for filtering out needed points.
        Input:
          - white_points: list of white points position in pixel coordinate (list of tuple (int(x), int(y)))
          - yellow_points: list of yellow points position in pixel coordinate (list of tuple (int(x), int(y)))
          - red_points: list of red points position in pixel coordinate (list of tuple (int(x), int(y)))
          - green_points: list of green points position in pixel coordinate (list of tuple (int(x), int(y)))
        Output:
          - white_points: list of filtered white points position in pixel coordinate (list of tuple (int(x), float(y)))
          - yellow_points: list of filtered yellow points position in pixel coordinate (list of tuple (int(x), float(y)))
          - red_points: list of filtered red points position in pixel coordinate (list of tuple (int(x), float(y)))
          - green_points: list of filtered green points position in pixel coordinate (list of tuple (int(x), float(y)))
        '''
        if len(white_points) <= self.anti_noise:
            white_points = []
        if len(yellow_points) <= self.anti_noise:
            yellow_points = []
        if len(green_points) <= self.anti_noise:
            green_points = []
        if len(red_points) <= self.anti_noise:
            red_points = []

        # filter yellow points
        if yellow_points:
            yellow_points = self.img_process.ransac_filter(yellow_points) # ransac line filter
        else:
            yellow_points = []


        # filter white points
        if white_points and not yellow_points:
            white_points = self.img_process.ransac_filter(white_points)


        # filter out the white points in the left side of yellow lines
        elif white_points and yellow_points:
            white_points_x, white_points_y = self.img_process.get_points_xy(white_points)
            yellow_points_x, yellow_points_y = self.img_process.get_points_xy(yellow_points)
            white_points_new = []
            for i in xrange(len(white_points)):
                if white_points_y[i] > np.max(yellow_points_y): # white points should always on the left side of yellow line
                    white_points_new.append(white_points[i])
            if white_points:
                white_points = self.img_process.ransac_filter(white_points_new)
            else:
                white_points = []
        elif not white_points:
            white_points = []


        # filter green points
        if green_points and not yellow_points and not white_points:
            green_points = self.img_process.get_bottom_green(green_points)
        elif green_points and yellow_points and not white_points:
            yellow_points_x, yellow_points_y = self.img_process.get_points_xy(yellow_points)
            green_points_x, green_points_y = self.img_process.get_points_xy(green_points)
            green_points_new = []
            for i in xrange(len(green_points)):
                if green_points_y[i] > np.mean(yellow_points_y):
                    green_points_new.append(green_points[i])
            green_points = green_points_new
            if green_points:
                green_points = self.img_process.get_bottom_green(green_points)
            else:
                green_points = []
        elif green_points and not yellow_points and white_points:
            white_points_x, white_points_y = self.img_process.get_points_xy(white_points)
            green_points_x, green_points_y = self.img_process.get_points_xy(green_points)
            green_points_new = []
            for i in xrange(len(green_points)):
                if green_points_y[i] < np.mean(white_points_y):
                    green_points_new.append(green_points[i])
            green_points = green_points_new
            if green_points:
                green_points = self.img_process.get_bottom_green(green_points)
            else:
                green_points = []
        elif green_points and yellow_points and white_points:
            white_points_x, white_points_y = self.img_process.get_points_xy(white_points)
            yellow_points_x, yellow_points_y = self.img_process.get_points_xy(yellow_points)
            green_points_x, green_points_y = self.img_process.get_points_xy(green_points)
            green_points_new = []
            for i in xrange(len(green_points)):
                if green_points_y[i] > np.mean(yellow_points_y) and green_points_y[i] < np.mean(white_points_y):
                    green_points_new.append(green_points[i])
            green_points = green_points_new
            if green_points:
                green_points = self.img_process.get_bottom_green(green_points)
            else:
                green_points = []
        elif not green_points:
            green_points = []


        # filter red points
        if red_points and not yellow_points and not white_points:
            red_points = self.img_process.get_bottom_red(red_points)
        elif red_points and yellow_points and not white_points:
            yellow_points_x, yellow_points_y = self.img_process.get_points_xy(yellow_points)
            red_points_x, red_points_y = self.img_process.get_points_xy(red_points)
            red_points_new = []
            for i in xrange(len(red_points)):
                if red_points_y[i] > np.mean(yellow_points_y):
                    red_points_new.append(red_points[i])
            red_points = red_points_new
            if red_points:
                red_points = self.img_process.get_bottom_red(red_points)
            else:
                red_points = []
        elif red_points and not yellow_points and white_points:
            white_points_x, white_points_y = self.img_process.get_points_xy(white_points)
            red_points_x, red_points_y = self.img_process.get_points_xy(red_points)
            red_points_new = []
            for i in xrange(len(red_points)):
                if red_points_y[i] < np.mean(white_points_y):
                    red_points_new.append(red_points[i])
            red_points = red_points_new
            if red_points:
                red_points = self.img_process.get_bottom_red(red_points)
            else:
                red_points = []
        elif red_points and yellow_points and white_points:
            white_points_x, white_points_y = self.img_process.get_points_xy(white_points)
            yellow_points_x, yellow_points_y = self.img_process.get_points_xy(yellow_points)
            red_points_x, red_points_y = self.img_process.get_points_xy(red_points)
            red_points_new = []
            for i in xrange(len(red_points)):
                if red_points_y[i] > np.mean(yellow_points_y) and red_points_y[i] < np.mean(white_points_y):
                    red_points_new.append(red_points[i])
            red_points = red_points_new
            if red_points:
                red_points = self.img_process.get_bottom_red(red_points)
            else:
                red_points = []
        elif not red_points:
            red_points = []


        return white_points, yellow_points, red_points, green_points


    def ground_projection(self, white_points, yellow_points, red_points, green_points):
        '''
        To perform ground projection, with addition of gaussian noise for red and green points for additional safety.
        Input:
          - white_points: list of filtered white points position in pixel coordinate (list of tuple (int(x), float(y)))
          - yellow_points: list of filtered yellow points position in pixel coordinate (list of tuple (int(x), float(y)))
          - red_points: list of filtered red points position in pixel coordinate (list of tuple (int(x), float(y)))
          - green_points: list of filtered green points position in pixel coordinate (list of tuple (int(x), float(y)))
        Output:
          - white_pos_x: list of x-position (float) in ground coordinate for white points
          - white_pos_y: list of y-position (float) in ground coordinate for white points
          - yellow_pos_x: list of x-position (float)in ground coordinate for yellow points
          - yellow_pos_y: list of y-position (float)in ground coordinate for yellow points
          - red_pos_x: list of x-position (float)in ground coordinate for red points
          - red_pos_y: list of y-position (float)in ground coordinate for red points
          - green_pos_x: list of x-position (float)in ground coordinate for green points
          - green_pos_y: list of y-position (float)in ground coordinate for green points
        '''
        
        # ground project yellow points
        if yellow_points:
            yellow_pos_x, yellow_pos_y = self.p2g.pixel2ground_local(yellow_points)
        else:
            yellow_pos_x = []
            yellow_pos_y = []


        # ground project white points
        if white_points:
            white_pos_x, white_pos_y = self.p2g.pixel2ground_local(white_points)
        else:
            white_pos_x = []
            white_pos_y = []


        # ground project red points
        if red_points:
            red_pos_x, red_pos_y = self.p2g.pixel2ground_local(red_points)
            red_pos_x, red_pos_y = self.p2g.add_gaussian_noise(red_pos_x, red_pos_y)
        else:
            red_pos_x = []
            red_pos_y = []


        # ground project green points
        if green_points:
            green_pos_x, green_pos_y = self.p2g.pixel2ground_local(green_points)
            green_pos_x, green_pos_y = self.p2g.add_gaussian_noise(green_pos_x, green_pos_y)
        else:
            green_pos_x = []
            green_pos_y = []


        return white_pos_x, white_pos_y, yellow_pos_x, yellow_pos_y, red_pos_x, red_pos_y, green_pos_x, green_pos_y


    def compute_follow_point_and_velocity(self, white_pos_x, white_pos_y, yellow_pos_x, yellow_pos_y, red_pos_x, red_pos_y, green_pos_x, green_pos_y):
        '''
        To compute follow point, velocity, and omega
        Input:
          - white_pos_x: list of white points x-position in ground coordinate (float)
          - white_pos_y: list of white points y-position in ground coordinate (float)
          - yellow_pos_x: list of yellow points x-position in ground coordinate (float)
          - yellow_pos_y: list of yellow points y-position in ground coordinate (float)
          - red_pos_x: list of red points x-position in ground coordinate (float)
          - red_pos_y: list of red points y-position in ground coordinate (float)
          - green_pos_x: list of green points x-position in ground coordinate (float)
          - green_pos_y: list of green points y-position in ground coordinate (float)
        Output:
          - follow_point_x: follow point x-position in ground coordinate (float)
          - follow_point_y: follow point y-position in ground coordinate (float)
          - v: linear velocity for duckiebot (float)
          - omega: angular velocity for duckiebot (float)
        '''
        

        if len(white_pos_x) == 0 and len(yellow_pos_x) > 0 and len(red_pos_x) == 0 and len(green_pos_x) == 0:
            print("Case #1: only see yellow line")
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            follow_point_x = np.mean(new_yellow_pos_x)
            follow_point_y = np.mean(new_yellow_pos_y) - self.bias
            v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_yellow)


        elif len(white_pos_x) > 0 and len(yellow_pos_x) == 0 and len(red_pos_x) == 0 and len(green_pos_x) == 0:
            print("Case #2: only see white line")
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            follow_point_x = np.mean(new_white_pos_x)
            follow_point_y = np.mean(new_white_pos_y) + self.bias
            v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_white)


        elif len(white_pos_x) > 0 and len(yellow_pos_x) > 0 and len(red_pos_x) == 0 and len(green_pos_x) == 0:
            print("Case #3: see both white and yellow lines")
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            follow_point_x = self.weight_white * np.mean(new_white_pos_x) + self.weight_yellow * np.mean(new_yellow_pos_x)
            follow_point_y = self.weight_white * np.mean(new_white_pos_y) + self.weight_yellow * np.mean(new_yellow_pos_y)
            v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_both)
            # bad_prediction = self.anti_bad_prediction.anti_white2yellow_prediction(new_yellow_pos_x,
            #                                                                         new_yellow_pos_y,
            #                                                                         new_white_pos_x,
            #                                                                         new_white_pos_y)
            # if bad_prediction:: # if bad prediction, depends on white line only
            #     print("-----Case #3.2: bad prediction")
            #     follow_point_x = np.mean(new_white_pos_x)
            #     follow_point_y = np.mean(new_white_pos_y) + self.bias
            #     v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_white)


        elif len(white_pos_x) == 0 and len(yellow_pos_x) > 0 and len(red_pos_x) > 0 and len(green_pos_x) == 0:
            print("Case #4: see yellow line and duckiebots")
            follow_point_y_list = []
            follow_point_x_list = []
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            x_red_min = np.min(red_pos_x)
            y_red_max = np.max(red_pos_y)
            y_red_min = np.min(red_pos_y) ##
            if self.avoid_obstacle:
                for i in xrange(len(new_yellow_pos_x)):
                    if new_yellow_pos_x[i] < x_red_min:
                        follow_point_y_list.append(new_yellow_pos_y[i] - self.bias)
                        follow_point_x_list.append(new_yellow_pos_x[i])
                    else:
                        d_left = new_yellow_pos_y[i] - y_red_max
                        d_right = abs(new_yellow_pos_y[i] - self.assumed_lane_width - y_red_min)
                        if d_left > 0.5 or d_right > 0.5: # only consider to calculate new follow point if we have enough opening
                            if d_left >= d_right: # choose left opening
                                follow_point_y_list.append(0.5 * (y_red_max + new_yellow_pos_y[i]))
                            else: # choose right opening
                                follow_point_y_list.append(0.5 * (new_yellow_pos_y[i] - self.assumed_lane_width - y_red_min))
                        follow_point_x_list.append(new_yellow_pos_x[i])
                        break # we do not want to consider any points that are located beyond where the duckiebot is
                follow_point_x = np.mean(follow_point_x_list)
                follow_point_y = np.mean(follow_point_y_list)
            else: # if not in obstacle avoidance mode, just keep going until gets too close to the obstacle
                follow_point_x = np.mean(new_yellow_pos_x)
                follow_point_y = np.mean(new_yellow_pos_y) - self.bias
            if follow_point_x > x_red_min:
                follow_point_x = x_red_min  # follow point should not go further beyond obstacle
            stop = self.spaceinfo.judge_stop(red_pos_x, red_pos_y, self.position_ground) # always check to stop if duckiebot is visible within lane
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_yellow)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0


        elif len(white_pos_x) == 0 and len(yellow_pos_x) > 0 and len(red_pos_x) == 0 and len(green_pos_x) > 0:
            print("Case #5: see yellow line and static obstacles")
            follow_point_y_list = []
            follow_point_x_list = []
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            if self.green_stop: # may not be needed for LFV challenge (better to turn off if we are sure there is no static obstacles)
                x_green_min = np.min(green_pos_x)
                y_green_max = np.max(green_pos_y)
                y_green_min = np.min(green_pos_y)
                if self.avoid_obstacle:
                    for i in xrange(len(new_yellow_pos_x)):
                        if new_yellow_pos_x[i] < x_green_min:
                            follow_point_y_list.append(new_yellow_pos_y[i] - self.bias)
                            follow_point_x_list.append(new_yellow_pos_x[i])
                        else:
                            d_left = new_yellow_pos_y[i] - y_green_max
                            d_right = abs(new_yellow_pos_y[i] - self.assumed_lane_width - y_green_min)
                            if d_left > 0.5 or d_right > 0.5: # only consider to calculate new follow point if we have enough opening
                                if d_left >= d_right: # choose left opening
                                    follow_point_y_list.append(0.5 * (y_green_max + new_yellow_pos_y[i]))
                                else: # choose right opening
                                    follow_point_y_list.append(0.5 * (new_yellow_pos_y[i] - self.assumed_lane_width - y_green_min))
                            follow_point_x_list.append(new_yellow_pos_x[i])
                            break # we do not want to consider any points that are located beyond where the duckiebot is
                    follow_point_x = np.mean(follow_point_x_list)
                    follow_point_y = np.mean(follow_point_y_list)
                else: # if not in obstacle avoidance mode, just keep going until gets too close to the obstacle
                    follow_point_x = np.mean(new_yellow_pos_x)
                    follow_point_y = np.mean(new_yellow_pos_y) - self.bias
                if follow_point_x > x_green_min:
                    follow_point_x = x_green_min  # follow point should not go over obstacle
                stop = self.spaceinfo.judge_stop(green_pos_x, green_pos_y, self.position_ground)
                if not stop:
                    v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_yellow)
                else:
                    print("-----Too close to duckiebot, STOP.")
                    v, omega = 0.0, 0.0
            else: # same as case #1 (only see yellow line)
                follow_point_x = np.mean(new_yellow_pos_x)
                follow_point_y = np.mean(new_yellow_pos_y) - self.bias
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_yellow)


        elif len(white_pos_x) == 0 and len(yellow_pos_x) > 0 and len(red_pos_x) > 0 and len(green_pos_x) > 0:
            print("Case #6: see yellow line, duckiebot, and static obstacles")
            follow_point_y_list = []
            follow_point_x_list = []
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            if self.green_stop:
                obs_list_x = green_pos_x + red_pos_x
                obs_list_y = green_pos_y + red_pos_y
            else:
                obs_list_x = red_pos_x
                obs_list_y = red_pos_y
            x_obs_min = np.min(obs_list_x)
            y_obs_max = np.max(obs_list_y)
            y_obs_min = np.min(obs_list_y) ##
            if self.avoid_obstacle:
                for i in xrange(len(new_yellow_pos_x)):
                    if new_yellow_pos_x[i] < x_obs_min:
                        follow_point_y_list.append(new_yellow_pos_y[i] - self.bias)
                        follow_point_x_list.append(new_yellow_pos_x[i])
                    else:
                        d_left = new_yellow_pos_y[i] - y_obs_max
                        d_right = abs(new_yellow_pos_y[i] - self.assumed_lane_width - y_obs_min)
                        if d_left > 0.5 or d_right > 0.5: # only consider to calculate new follow point if we have enough opening
                            if d_left >= d_right: # choose left opening
                                follow_point_y_list.append(0.5 * (y_obs_max + new_yellow_pos_y[i]))
                            else: # choose right opening
                                follow_point_y_list.append(0.5 * (new_yellow_pos_y[i] - self.assumed_lane_width - y_obs_min))
                        follow_point_x_list.append(new_yellow_pos_x[i])
                        break # we do not want to consider any points that are located beyond where the duckiebot is
                follow_point_x = np.mean(follow_point_x_list)
                follow_point_y = np.mean(follow_point_y_list)
            else: # if not in obstacle avoidance mode, just keep going until gets too close to the obstacle
                follow_point_x = np.mean(new_yellow_pos_x)
                follow_point_y = np.mean(new_yellow_pos_y) - self.bias
            if follow_point_x > x_obs_min:
                follow_point_x = x_obs_min  # follow point should not go over obstacle
            stop = self.spaceinfo.judge_stop(obs_list_x, obs_list_y, self.position_ground) # always check to stop if duckiebot is visible within lane
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_yellow)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0


        elif len(white_pos_x) > 0 and len(yellow_pos_x) == 0 and len(red_pos_x) > 0 and len(green_pos_x) == 0:
            print("Case #7: see white line and duckiebot")
            follow_point_y_list = []
            follow_point_x_list = []
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            x_red_min = np.min(red_pos_x)
            y_red_max = np.max(red_pos_y)
            y_red_min = np.min(red_pos_y) ##
            if self.avoid_obstacle:
                for i in xrange(len(new_white_pos_x)):
                    if new_white_pos_x[i] < x_red_min:
                        follow_point_y_list.append(new_white_pos_y[i] + self.bias)
                        follow_point_x_list.append(new_white_pos_x[i])
                    else:
                        d_right = abs(new_white_pos_y[i] - y_red_min)
                        d_left = abs(new_white_pos_y[i] + self.assumed_lane_width - y_red_max)
                        if d_left > 0.5 or d_right > 0.5: # only consider to calculate new follow point if we have enough opening
                            if d_left >= d_right: # choose left opening
                                follow_point_y_list.append(0.5 * (new_white_pos_y[i] + self.assumed_lane_width + y_red_max))
                            else: # choose right opening
                                follow_point_y_list.append(0.5 * (y_red_min + new_white_pos_y[i]))
                        follow_point_x_list.append(new_white_pos_x[i])
                        break
                follow_point_x = np.mean(follow_point_x_list)
                follow_point_y = np.mean(follow_point_y_list)
            else: # if not in obstacle avoidance mode, just keep going until gets too close to the obstacle
                follow_point_x = np.mean(new_white_pos_x)
                follow_point_y = np.mean(new_white_pos_y) + self.bias
            if follow_point_x > x_red_min:
                follow_point_x = x_red_min  # fixing follow point while unexpected computing error happened
            stop = self.spaceinfo.judge_stop(red_pos_x, red_pos_y, self.position_ground) # always check to stop if duckiebot is visible within lane
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_white)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0


        elif len(white_pos_x) > 0 and len(yellow_pos_x) == 0 and len(red_pos_x) == 0 and len(green_pos_x) > 0:
            print("Case #8: see white line and static obstacles")
            follow_point_y_list = []
            follow_point_x_list = []
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            if self.green_stop: # may not be needed for LFV challenge (better to turn off if we are sure there is no static obstacles)
                x_green_min = np.min(green_pos_x)
                y_green_min = np.min(green_pos_y)
                y_green_max = np.max(green_pos_y)
                if self.avoid_obstacle:
                    for i in xrange(len(new_white_pos_x)):
                        if new_white_pos_x[i] < x_red_min:
                            follow_point_y_list.append(new_white_pos_y[i] + self.bias)
                            follow_point_x_list.append(new_white_pos_x[i])
                        else:
                            d_right = abs(new_white_pos_y[i] - y_red_min)
                            d_left = abs(new_white_pos_y[i] + self.assumed_lane_width - y_red_max)
                            if d_left > 0.5 or d_right > 0.5: # only consider to calculate new follow point if we have enough opening
                                if d_left >= d_right: # choose left opening
                                    follow_point_y_list.append(0.5 * (new_white_pos_y[i] + self.assumed_lane_width + y_red_max))
                                else: # choose right opening
                                    follow_point_y_list.append(0.5 * (y_red_min + new_white_pos_y[i]))
                            follow_point_x_list.append(new_white_pos_x[i])
                            break
                    follow_point_x = np.mean(follow_point_x_list)
                    follow_point_y = np.mean(follow_point_y_list)
                else: # if not in obstacle avoidance mode, just keep going until gets too close to the obstacle
                    follow_point_x = np.mean(new_white_pos_x)
                    follow_point_y = np.mean(new_white_pos_y) + self.bias
                if follow_point_x > x_green_min:
                    follow_point_x = x_green_min
                stop = self.spaceinfo.judge_stop(green_pos_x, green_pos_y, self.position_ground)
                if not stop:
                    v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_white)
                else:
                    print("-----Too close to duckiebot, STOP.")
                    v, omega = 0.0, 0.0
            else: # same as case #2 (only see white line)
                follow_point_x = np.mean(new_white_pos_x)
                follow_point_y = np.mean(new_white_pos_y) + self.bias
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_white)


        elif len(white_pos_x) > 0 and len(yellow_pos_x) == 0 and len(red_pos_x) > 0 and len(green_pos_x) > 0:
            print("Case #9: see white line, duckiebots, and static obstacles")
            follow_point_y_list = []
            follow_point_x_list = []
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            if self.green_stop:
                obs_list_x = green_pos_x + red_pos_x
                obs_list_y = green_pos_y + red_pos_y
            else:
                obs_list_x = red_pos_x
                obs_list_y = red_pos_y
            x_obs_min = np.min(obs_list_x)
            y_obs_min = np.min(obs_list_y)
            y_obs_max = np.max(obs_list_y)
            if self.avoid_obstacle:
                for i in xrange(len(new_white_pos_x)):
                    if new_white_pos_x[i] < x_obs_min:
                        follow_point_y_list.append(new_white_pos_y[i] + self.bias)
                        follow_point_x_list.append(new_white_pos_x[i])
                    else:
                        d_left = abs(new_white_pos_y[i] - y_obs_min)
                        d_right = abs(new_white_pos_y[i] + self.assumed_lane_width - y_obs_max)
                        if d_left > 0.5 or d_right > 0.5: # only consider to calculate new follow point if we have enough opening
                            if d_left >= d_right: # choose left opening
                                follow_point_y_list.append(0.5 * (new_white_pos_y[i] + self.assumed_lane_width + y_obs_max))
                            else: # choose right opening
                                follow_point_y_list.append(0.5 * (y_obs_min + new_white_pos_y[i]))
                        follow_point_x_list.append(new_white_pos_x[i])
                        break
                follow_point_x = np.mean(follow_point_x_list)
                follow_point_y = np.mean(follow_point_y_list)
            else: # if not in obstacle avoidance mode, just keep going until gets too close to the obstacle
                follow_point_x = np.mean(new_white_pos_x)
                follow_point_y = np.mean(new_white_pos_y) + self.bias
            if follow_point_x > x_obs_min:
                follow_point_x = x_obs_min
            stop = self.spaceinfo.judge_stop(obs_list_x, obs_list_y, self.position_ground)
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_white)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0
                

        elif len(white_pos_x) > 0 and len(yellow_pos_x) > 0 and len(red_pos_x) > 0 and len(green_pos_x) == 0:
            print("Case #10: see yellow line, white line, duckiebots")
            follow_point_y_list = []
            follow_point_x_list = []
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            x_red_min = np.min(red_pos_x)
            y_red_max = np.max(red_pos_y)
            y_red_min = np.min(red_pos_y) ##
            if self.avoid_obstacle:
                pass
            else: # if not in obstacle avoidance mode, just keep going until it gets too close to the obstacle
                follow_point_x = self.weight_white * np.mean(new_white_pos_x) + self.weight_yellow * np.mean(new_yellow_pos_x)
                follow_point_y = self.weight_white * np.mean(new_white_pos_y) + self.weight_yellow * np.mean(new_yellow_pos_y)
            if follow_point_x > x_red_min:
                follow_point_x = x_red_min
            stop = self.spaceinfo.judge_stop(red_pos_x, red_pos_y, self.position_ground) # always check to stop if duckiebot is visible within lane
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_both)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0


        elif len(white_pos_x) > 0 and len(yellow_pos_x) > 0 and len(red_pos_x) == 0 and len(green_pos_x) > 0:
            print("Case #11: see white line, yellow line, and static obstacles")
            follow_point_y_list = []
            follow_point_x_list = []
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            if self.green_stop:
                x_green_min = np.min(green_pos_x)
                y_green_max = np.max(green_pos_y)
                y_green_min = np.min(green_pos_y) ##
                if self.avoid_obstacle:
                    pass
                else: # if not in obstacle avoidance mode, just keep going until it gets too close to the obstacle
                    follow_point_x = self.weight_white * np.mean(new_white_pos_x) + self.weight_yellow * np.mean(new_yellow_pos_x)
                    follow_point_y = self.weight_white * np.mean(new_white_pos_y) + self.weight_yellow * np.mean(new_yellow_pos_y)
                if follow_point_x > x_green_min:
                    follow_point_x = x_green_min
            else: # if does not care about static obstacles, treat it as only see white and yellow lines
                follow_point_x = self.weight_white * np.mean(new_white_pos_x) + self.weight_yellow * np.mean(new_yellow_pos_x)
                follow_point_y = self.weight_white * np.mean(new_white_pos_y) + self.weight_yellow * np.mean(new_yellow_pos_y)
            stop = self.spaceinfo.judge_stop(red_pos_x, red_pos_y, self.position_ground) # always check to stop if duckiebot is visible within lane
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_both)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0


        elif len(white_pos_x) > 0 and len(yellow_pos_x) > 0 and len(green_pos_x) > 0 and len(green_pos_x) > 0:
            print("Case #12: see white line, yellow line, duckiebots, and static obstacles")
            follow_point_y_list = []
            follow_point_x_list = []
            new_white_pos_x, new_white_pos_y = self.img_process.normalize_new_pos(white_pos_x, white_pos_y)
            new_yellow_pos_x, new_yellow_pos_y = self.img_process.normalize_new_pos(yellow_pos_x, yellow_pos_y)
            if self.green_stop:
                obs_list_x = green_pos_x + red_pos_x
                obs_list_y = green_pos_y + red_pos_y
            else:
                obs_list_x = red_pos_x
                obs_list_y = red_pos_y
            x_obs_min = np.min(obs_list_x)
            y_obs_max = np.max(obs_list_y)
            y_obs_min = np.min(obs_list_y) ##
            if self.avoid_obstacle:
                pass
            else: # if not in obstacle avoidance mode, just keep going until it gets too close to the obstacle
                follow_point_x = self.weight_white * np.mean(new_white_pos_x) + self.weight_yellow * np.mean(new_yellow_pos_x)
                follow_point_y = self.weight_white * np.mean(new_white_pos_y) + self.weight_yellow * np.mean(new_yellow_pos_y)
            if follow_point_x > x_obs_min:
                follow_point_x = x_obs_min
            stop = self.spaceinfo.judge_stop(obs_list_x, obs_list_y, self.position_ground) # always check to stop if duckiebot is visible within lane
            if not stop:
                v, omega = self.getSpeed(follow_point_x, follow_point_y, self.k_omega_yellow)
            else:
                print("-----Too close to duckiebot, STOP.")
                v, omega = 0.0, 0.0
                    
        
        else:
            print("Case #13: do not see lines")
            if self.green_stop:
                if red_pos_x or green_pos_x:
                    print("------Case #13.1: see obstacles, stop")
                    v, omega = 0.0, 0.0
                    follow_point_x, follow_point_y = 0.0, 0.0
                else:
                    print("------Case #13.1: no obstacles, move slowly")
                    v, omega = 0.1, 0.1
                    follow_point_x, follow_point_y = 0.0, 0.0
            else:
                if red_pos_x:
                    print("------Case #13.1: see obstacles, stop")
                    v, omega = 0.0, 0.0
                    follow_point_x, follow_point_y = 0.0, 0.0
                else:
                    print("------Case #13.1: no obstacles, move slowly")
                    v, omega = 0.1, 0.1
                    follow_point_x, follow_point_y = 0.0, 0.0
        return follow_point_x, follow_point_y, v, omega


    def callback(self, data):
        '''
        This callback gets called everytime we receive a new segmentation map.
        Input:
          - data: a segmentation map (CompressedImage)
        Output:
          - wd_data: message containing velocity and omega values (Twist2DStamped)
          - Optional: msg_img_ran, ground projection visualization images(Compressed (ros_msg))
          - Optional: msg_img_gp, ransac filter visualization images(Compressed (ros_msg))
        '''
        

        img = self.rosCompressedImageToArray(data.data)
        img = self.img_process.cut_down_top(img) # crop image
        white_points, yellow_points, _, red_points, green_points = self.img_process.get_pixel_position(img) # return position of each points in pixel coordinate
        white_points, yellow_points, red_points, green_points = self.point_filter(white_points, yellow_points, red_points, green_points) # apply noise filter
        white_pos_x, white_pos_y, yellow_pos_x, yellow_pos_y, red_pos_x, red_pos_y, green_pos_x, green_pos_y = self.ground_projection(white_points, yellow_points, red_points, green_points)
        follow_point_x, follow_point_y, v, omega = self.compute_follow_point_and_velocity(white_pos_x, white_pos_y, yellow_pos_x, yellow_pos_y, red_pos_x, red_pos_y, green_pos_x, green_pos_y)


        # Visualize RANSAC and ground projection
        if self.enable_visualization:
            msg_img_ran = self.visual.visualize_ransac(white_points, yellow_points)
            try:
                self.viz_ransac_pub.publish(msg_img_ran)
            except CvBridgeError as e:
                rospy.logerr(str(e))
            msg_img_gp = self.visual.visualize_ground_projection_and_follow(white_pos_x, white_pos_y,
                                                                            yellow_pos_x, yellow_pos_y,
                                                                            red_pos_x, red_pos_y,
                                                                            green_pos_x, green_pos_y,
                                                                            follow_point_x, follow_point_y)
            try:
                self.viz_ground_project_pub.publish(msg_img_gp)
            except CvBridgeError as e:
                rospy.logerr(str(e))


        # Publish v and omega
        wd_data = Twist2DStamped()
        wd_data.v = v
        wd_data.omega = omega
        self.pub.publish(wd_data)


    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = LFVController()
        node.spin()
    except rospy.ROSInterruptException:
        pass
