#! /usr/bin/env python
import math
import numpy as np


class SpaceInfo:
    def __init__(self):
        self.info = "compute space info"

    
    def get_p2l_distance(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Calculate distance between a point and a straight line.
        Input:
          - pointX: x-position in ground coordinate (float)
          - pointY: y-position in ground coordinate (float)
          - lineX1: x-position of point 1 on this straight line (float)
          - lineY1: y-position of point 1 on this straight line (float)
          - lineX2: x-position of point 2 on this straight line (float)
          - lineY2: y-position of point 2 on this straight line (float)
        Output:
          - dis: the distance between the point and the line (float)
        '''
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        try:
            dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))
        except ZeroDivisionError: # point is on the line, return point to point distance
            dis = np.sqrt((pointX - lineX1) ** 2 + (pointY - lineY1) ** 2)
        return dis


    def get_slope(self, lineX1_, lineY1_, lineX2_, lineY2_):
        '''
        Calculate the slope of a straight line.
        Input:
          - lineX1_: x-position of point 1 on this straight line (float)
          - lineY1_: y-position of point 1 on this straight line (float)
          - lineX2_: x-position of point 2 on this straight line (float)
          - lineY2_: y-position of point 2 on this straight line (float)
        Output:
          - slope: the slope of the line (float)
        '''
        dx = float(lineX1_ - lineX2_)
        dy = float(lineY1_ - lineY2_)
        if dx != 0:
            slope = dy / dx
        else:
            slope = 1e5
        return slope


    def get_p2p_distance(self, x1, y1, x2, y2):
        '''
        To calculate distance between two points
        Input:
          - x1: x-position of point 1 (float)
          - y1: y-position of point 1 (float)
          - x2: x-position of point 2 (float)
          - y2: y-position of point 2 (float)
        Output:
          - d: the distance between two points (float)
        '''
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return d


    def judge_stop(self, pos_x, pos_y, min_distance = 0.4):
        '''
        Function to judge if Duckiebot needs to stop.
        Input:
          - pos_x: list of red/green x-position(float) in ground coordinate
          - pos_y: list of red/green y-position(float) in ground coordinate
        Output:
          - stop: True if duckiebot needs to stop(bool)
        '''
        stop = False
        distance_list = []
        for i in xrange(len(pos_x)):
            distance_ = self.get_p2p_distance(pos_x[i], pos_y[i], 0.0, 0.0)
            distance_list.append(distance_)
        dis_safe = np.min(distance_list)
        if dis_safe <= min_distance:
            stop = True
        return stop


class AntiBadPrediction:
    def __init__(self):
        self.info = "anti_bad_prediction"


    def get_p2l_distance(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Calculate distance between a point and a straight line.
        Input:
          - pointX: x-position in ground coordinate (float)
          - pointY: y-position in ground coordinate (float)
          - lineX1: x-position of point 1 on this straight line (float)
          - lineY1: y-position of point 1 on this straight line (float)
          - lineX2: x-position of point 2 on this straight line (float)
          - lineY2: y-position of point 2 on this straight line (float)
        Output:
          - dis: the distance between the point and the line (float)
        '''
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        try:
            dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))
        except ZeroDivisionError:
            dis = np.sqrt((pointX - lineX1) ** 2 + (pointY - lineY1) ** 2)
        return dis


    def anti_white2yellow_prediction(self, new_yellow_pos_x, new_yellow_pos_y, new_white_pos_x, new_white_pos_y):
        '''
        To judge if segmentation model mistakenly predict some points belonging to white lines as yellow.
        Input:
          - new_white_pos_x: list of white points x-position(float) in ground coordinate
          - new_white_pos_y: list of white points y-position(float) in ground coordinate
          - new_yellow_pos_x: list of yellow points x-position(float) in ground coordinate
          - new_yellow_pos_y: list of yellow points y-position(float) in ground coordinate
        Output:
          - bad_prediction: Boolean flag, True if bad prediction is detected
        '''
        bad_prediction = False
        c = 0
        for i in range(len(new_yellow_pos_x)):
            if self.get_p2l_distance(new_yellow_pos_x[i], new_yellow_pos_y[i], new_white_pos_x[0], new_white_pos_y[0],
                                     new_white_pos_x[1], new_white_pos_y[1]) < 0.1:
                c += 1
            if c > int(0.5 * len(new_yellow_pos_x)): # if half of white and yellow pixels distances are too close
                bad_prediction = True
                break
        return bad_prediction
