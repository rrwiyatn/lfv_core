#! /usr/bin/env python
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import cv2


class ImgPixelsProcess:
    def __init__(self):
        self.width = 160 # desired image width to be considered
        self.height = 80 # desired image height to be considered
        self.original_height = 120
        self.cut_down_height = self.original_height - self.height
        self.white_pixel = np.array([1,1,1])
        self.yellow_pixel = np.array([1,1,0])
        self.purple_pixel = np.array([1,0,1])
        self.red_pixel = np.array([1,0,0])
        self.green_pixel = np.array([0,1,0])
        self.ransac = linear_model.RANSACRegressor()

    
    def cut_down_top(self, img):
        '''
        This function is for cropping the top part of input image (120, 160) --> (80, 160)
        Input:
          - img: an RGB image array (ndarray)
        Output:
          - img: an RGB image array (ndarray)
        '''
        img = img[self.cut_down_height:self.original_height, 0:self.width]
        return img

    
    def get_points_xy(self, points):
        '''
        This function is for returning x and y position for a point (x, y) as two different lists
        Input:
          - points: list of points position in pixel coordinate (list of tuple (int(x), int/float(y)))
        Output:
          - x_list: list of points x-position(int) in pixel coordinate
          - y_list: list of points y-position(int/float) in pixel coordinate
        '''
        x_list = []
        y_list = []
        for i in points:
            (v, u) = (i[0], i[1])
            x_list.append(v)
            y_list.append(u)
        return x_list, y_list


    def get_pixel_position(self, img):
        '''
        Function is to obtain lists of points that correspond to each class (white, yellow, purple, red, green).
        For LFV challenge, we can ignore the purple and green points.
        Input:
          - img: an RGB image array (ndarray)
        Output:
          - white_points: list of white points position in pixel coordinate (list of tuple (int(x), float(y)))
          - yellow_points: list of yellow points position in pixel coordinate (list of tuple (int(x), float(y)))
          - purple_points: list of purple points position in pixel coordinate (list of tuple (int(x), float(y)))
          - red_points: list of red points position in pixel coordinate (list of tuple (int(x), int(y)))
          - green_points: list of  green points position in pixel coordinate (list of tuple (int(x), int(y)))
        '''
        im = np.round(img)
        white_mask = cv2.inRange(im,self.white_pixel,self.white_pixel) / 255.
        yellow_mask = cv2.inRange(im, self.yellow_pixel, self.yellow_pixel) / 255.
        green_mask = cv2.inRange(im, self.green_pixel, self.green_pixel) / 255.
        red_mask = cv2.inRange(im, self.red_pixel, self.red_pixel) / 255.
        purple_mask = cv2.inRange(im, self.purple_pixel, self.purple_pixel) / 255.
        white_idxs = np.asarray(np.nonzero(white_mask))
        yellow_idxs = np.asarray(np.nonzero(yellow_mask))
        green_idxs = np.asarray(np.nonzero(green_mask))
        red_idxs = np.asarray(np.nonzero(red_mask))
        purple_idxs = np.asarray(np.nonzero(purple_mask))
        white_idxs = np.swapaxes(white_idxs, 0, 1)
        yellow_idxs = np.swapaxes(yellow_idxs, 0, 1)
        green_idxs = np.swapaxes(green_idxs, 0, 1)
        red_idxs = np.swapaxes(red_idxs, 0, 1)
        purple_idxs = np.swapaxes(purple_idxs, 0, 1)
        white_points = list(white_idxs)
        yellow_points = list(yellow_idxs)
        green_points = list(green_idxs)
        red_points = list(red_idxs)
        purple_points = list(purple_idxs)
        return white_points, yellow_points, purple_points, red_points, green_points


    def get_bottom_green(self, green_points):
        '''
        Function to select only the bottom part of green points.
        Input:
          - green_points:  green points position in pixel coordinate (list of tuple (int(x), int(y)))
        Output:
          - green_points_selected: green points position in pixel coordinate (list of tuple (int(x), int(y)))
        '''
        green_x = []
        green_y = []
        green_points_selected = []
        for i in green_points:
            (u, v) = (i[1], i[0])
            green_x.append(v)
            green_y.append(u)
        green_x_max = np.max(green_x)
        green_y_max = np.max(green_y)
        green_y_min = np.min(green_y)
        n = 0
        for i in range(green_y_min, green_y_max + 1):
            if n % 5 == 0:
                green_points_selected.append((green_x_max, i))
            n += 1
        return green_points_selected


    def get_bottom_red(self, red_points):
        '''
        Function to select only the bottom part of red points.
        Input:
          - red_points: red points position in pixel coordinate (list of tuple (int(x), int(y)))
        Output:
          - red_points_selected: green points position in pixel coordinate (list of tuple (int(x), int(y)))
        '''
        red_x = []
        red_y = []
        red_points_selected = []
        for i in red_points:
            (u, v) = (i[1], i[0])
            red_x.append(v)
            red_y.append(u)
        red_x_max = np.max(red_x)
        red_y_max = np.max(red_y)
        red_y_min = np.min(red_y)
        n = 0
        for i in range(red_y_min, red_y_max + 1):
            if n % 5 == 0:
                red_points_selected.append((red_x_max, i))
            n += 1
        return red_points_selected


    def ransac_filter(self, pos):
        '''
        Function to fit the points with RANSAC.
        Input:
          - pos: points position in pixel coordinate (list of tuple (int(x), int(y)))
        Output:
          - new_points_list: points, in a straight line, position in pixel coordinate (list of tuple (int(x), float(y)))
        '''
        try:
            x_list = []
            y_list = []
            new_points_list = []
            new_point_ = new_points_list.append
            for i in pos:
                (x, y) = i
                x_list.append(x)
                y_list.append(y)
            x_array = np.array(x_list).reshape(len(x_list), 1)
            y_array = np.array(y_list).reshape(len(y_list), 1)
            self.ransac.fit(x_array, y_array)
            line_X = np.arange(x_array.min(), x_array.max())[:, np.newaxis]
            line_y_ransac = self.ransac.predict(line_X)
            new_x_array = np.array(line_X).reshape(line_X.shape[0], )
            new_y_array = np.array(line_y_ransac).reshape(line_y_ransac.shape[0], )
            new_x_list = new_x_array.tolist()
            new_y_list = new_y_array.tolist()
            for i in range(len(new_x_list)): # remove out of range pixels
                if new_x_list[i] < 80 and new_x_list[i] > 0:
                   if new_y_list[i] < 160 and new_y_list[i] > 0:
                        x = new_x_list[i]
                        y = new_y_list[i]
                        new_point_((x,y))
        except ValueError:
            new_points_list = []
        return new_points_list


    def normalize_new_pos(self, pos_x, pos_y):
        '''
        Function to sample 10 groups of points along a straight line
        Input:
          - pos_x: list of x-position (float) in ground coordinate
          - pos_y: list of y-position (float) in ground coordinate
        Output:
          - pos_xs: list of normalized x-position (float) in ground coordinate
          - pos_ys: list of normalized y-position (float) in ground coordinate
        '''
        pos_xs = []
        pos_ys = []
        for i in range(1, 10):
            new_pos_x = float(i) / 10.0 * 2.0 * np.mean(pos_x)
            new_pos_y = float(i) / 10.0 * 2.0 * np.mean(pos_y)
            pos_xs.append(new_pos_x)
            pos_ys.append(new_pos_y)
        return pos_xs, pos_ys

    
    '''EXPERIMENTAL'''
    def normalize_new_pos_new(self, pos_x, pos_y):# sample points based on order
        '''This function is for sampleing 10 groups of points on a curve line based on order'''
        # Input:
        #   - pos_x, x position in ground coordinate for points(list, element type: float)
        #   - pos_y, y position in ground coordinate for points(list, element type: float)
        # Output:
        #   - pos_xs, x position in ground coordinate for points(list, element type: float)
        #   - pos_ys, y position in ground coordinate for points(list, element type: float)
        # This function is working with function ransac_cubic_filter, NOT USED
        pos_xs = []
        pos_ys = []
        x_append = pos_xs.append
        y_append = pos_ys.append
        for i in range(10):
            new_pos_x = pos_x[int(((9-i)/10.0) * len(pos_x))]
            new_pos_y = pos_y[int(((9-i)/10.0) * len(pos_x))]
            x_append(new_pos_x)
            y_append(new_pos_y)
        return pos_xs, pos_ys


    def ransac_cubic_filter(self, pos):
        '''This function is for selecting the point on a curve line, filtering a line with a group of points'''
        # Input:
        #   - pos,  points position in pixel coordinate(list, element type tuple, (x, y), x(int), y(int))
        # Output:
        #   - new_points_list,  points, in a curve line, position in pixel coordinate (list, element type tuple, (x, y), x(int), y(float))
        # This function is not meet expected performance, work with function normalize_new_pos_new, not used.
        x_list = []
        y_list = []
        old_x = x_list.append
        old_y = y_list.append
        new_points_list = []
        new_point_ = new_points_list.append
        # regr = LinearRegression()
        cubic = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)
        for i in pos:
            (x, y) = i
            old_x(80-x)
            old_y(y)
        x_max = np.max(x_list)
        x_min = np.min(x_list)
        y_max = np.max(y_list)
        y_min = np.min(y_list)
        x_array = np.array(x_list).reshape(len(x_list), 1)
        y_array = np.array(y_list).reshape(len(y_list), 1)
        try:
            x_cubic = cubic.fit_transform(x_array)
            x_cubic_fit = np.arange(x_cubic.min(), x_cubic.max(), 1)[:, np.newaxis]
            ransac = self.ransac.fit(x_cubic, y_array)
            y_cubic_fit = ransac.predict(cubic.fit_transform(x_cubic_fit))
            x_cubic_fit = x_cubic_fit.reshape(len(x_cubic_fit),)
            y_cubic_fit = y_cubic_fit.reshape(len(y_cubic_fit),)

            new_x_list = x_cubic_fit.tolist()
            new_y_list = y_cubic_fit.tolist()
            for i in range(len(new_x_list)):
                if new_x_list[i] < x_max and new_x_list[i] > x_min:
                    if new_y_list[i] < y_max and new_y_list[i] > y_min:
                        x = 80 - new_x_list[i]
                        y = new_y_list[i]
                        new_point_((x, y))
        except ValueError:
            new_points_list = []
        return new_points_list
