# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2


class OptimalControl(object):
    """Optimal Control (Forecasting)"""
    def __init__(self, visualize=True):
        self.w           = None
        self.feature_map = []
        self.reward      = []
        self.V           = []
        self.pax         = []
        self.D           = []
        self.img         = []
        self.start       = []
        self.end         = []
        self.VISUALIZE   = visualize

    def load_terminal_pts(self, input_filename):
        print "load terminal points ..."
        with open(input_filename) as f:
            s = map(int, f.readline().strip().split(' '))
            e = map(int, f.readline().strip().split(' '))
            self.start = np.array([s[1], s[0]], dtype='int')
            self.end = np.array([e[1], e[0]], dtype='int')
        print "    start:", self.start
        print "    end:", self.end

    def load_reward_weights(self, input_filename):
        print "load reward weights ..."
        self.w = np.loadtxt(input_filename)
        print "    number of weights loaded:", self.w.shape[0]

    def load_features(self, input_filename):
        print "load features ..."
        self.feature_map = np.load(input_filename)
        print "    number of feature maps loaded:", self.feature_map.shape[0]

    def load_image(self, input_filename):
        print "load image ..."
        img_orig = cv2.imread(input_filename)
        self.img = cv2.resize(img_orig, (self.feature_map[0].shape[1], self.feature_map[0].shape[0]))
        if self.VISUALIZE:
            cv2.imshow("Bird's Eye Image", self.img)
            cv2.waitKey(100)

    def set_named_window(self):
        if self.VISUALIZE:
            cv2.namedWindow("Bird's Eye Image");
            cv2.namedWindow("Reward Function");
            cv2.namedWindow("MaxEnt Value Function");
            cv2.namedWindow("Forecast Distribution");
            cv2.moveWindow("Bird's Eye Image", 0,0);
            cv2.moveWindow("Reward Function", self.img.shape[1], 0);
            cv2.moveWindow("MaxEnt Value Function", 0, 261);
            cv2.moveWindow("Forecast Distribution", self.img.shape[1], 261);

    def compute_value_function(self, output_filename):
        print "compute value function (modified; faster) ..."

        # compute reward
        self.reward = np.zeros(self.feature_map[0].shape, dtype='float')
        for i in range(self.w.shape[0]):
            self.reward += self.w[i] * self.feature_map[i]

        if self.VISUALIZE:
            dsp = self.__color_map(self.reward)
            cv2.imshow("Reward Function", dsp)
            cv2.waitKey(100)

        self.V = np.ones(self.feature_map[0].shape, dtype='float') * -sys.float_info.max
        self.V[self.end[0], self.end[1]] = 0.0
        V = self.V.copy()

        n = 0
        while True:
            v = self.V.copy() * 1.0
            V_padded = cv2.copyMakeBorder(v, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                          0, -sys.float_info.max)
            V_padded *= 1.0

            sub_array = np.zeros([9, v.shape[0], v.shape[1]], dtype='float')

            sub_arr_index = 0
            for y in range(3):
                for x in range(3):
                    sub_array[sub_arr_index, :, :] = V_padded[y:y+v.shape[0],
                                                              x:x+v.shape[1]].copy()
                    sub_arr_index += 1

            is_neg_inf = sub_array == -sys.float_info.max
            neg_inf_mat = np.logical_not(np.prod(is_neg_inf, axis=0))

            for i, sub_elem in enumerate(sub_array):
                if i == 4:
                    continue
                minv = np.minimum(self.V, sub_elem)
                maxv = np.maximum(self.V, sub_elem)
                softmax = maxv + np.log(1.0 + np.exp(minv - maxv))
                self.V[neg_inf_mat] = softmax[neg_inf_mat]

            self.V[neg_inf_mat] += self.reward[neg_inf_mat]

            self.V[self.end[0], self.end[1]] = 0.0

            # convergence criteria
            residual = cv2.absdiff(self.V, V)
            minVal, maxVal = cv2.minMaxLoc(residual)[0:2]
            V = self.V.copy()

            if maxVal < 0.9:
                break

            if self.VISUALIZE:
                dst = self.__color_map(self.V)
                dst = cv2.addWeighted(self.img, 0.5, dst, 0.5, 0)
                cv2.imshow("MaxEnt Value Function", dst)
                cv2.waitKey(1)

            n += 1
            if n > 1000:
                print "ERROR: max number of iterations"
                sys.exit(-1)

        print "converged in %d steps" % n

        # output value function
        np.save(output_filename, self.V)

    def compute_policy(self, output_filename):
        print "compute policy"
        na = 9

        for a in range(na):
            self.pax.append(np.zeros(self.V.shape, dtype='float'))

        V_padded = cv2.copyMakeBorder(self.V, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0, -np.inf)

        for col in range(V_padded.shape[0]-2):
            for row in range(V_padded.shape[1]-2):
                sub = V_padded[col:col+3, row:row+3]
                minVal, maxVal = cv2.minMaxLoc(sub)[0:2]
                p = sub - maxVal
                p = np.exp(p)
                p[1,1] = 0
                su = np.sum(p)
                if su > 0:
                    p /= su
                else:
                    p = 1.0 / (na - 1.0)
                p = p.flatten()
                for a in range(na):
                    self.pax[a][col, row] = p[a]

        # output
        np.save(output_filename, self.pax)

    def compute_forecast_dist(self, output_image_filename, output_prob_filename):
        print "compute forecast distribution (modified; faster)  ..."
        self.D = np.zeros(self.V.shape, dtype='float')
        N = [self.D.copy(), self.D.copy()]
        N[0][self.start[0], self.start[1]] = 1.0
        col = N[0].shape[0]
        row = N[0].shape[1]

        border = self.__make_border_mask(N[0].shape)
        n = 0
        while True:
            N[1] *= 0

            mask = np.zeros(N[0].shape, dtype=np.bool)
            mask[N[0] > sys.float_info.min] = True
            mask[self.end[0], self.end[1]] = False
            padded_mask = np.lib.pad(mask, 1, 'constant', constant_values=False)

            N_pax_tmp = []
            for i in range(9):
                 N_pax_tmp.append(N[0] * self.pax[i])

            # north-west (top-left)
            N[1][padded_mask[2:2 + col, 2:2 + row]] += N_pax_tmp[0][np.logical_and(mask, border[0])]
            # north (top)
            N[1][padded_mask[2:2 + col, 1:1 + row]] += N_pax_tmp[1][np.logical_and(mask, border[1])]
            # north-east (top-right)
            N[1][padded_mask[2:2 + col, 0:0 + row]] += N_pax_tmp[2][np.logical_and(mask, border[2])]
            # west (left)
            N[1][padded_mask[1:1 + col, 2:2 + row]] += N_pax_tmp[3][np.logical_and(mask, border[3])]
            # east (right)
            N[1][padded_mask[1:1 + col, 0:0 + row]] += N_pax_tmp[5][np.logical_and(mask, border[5])]
            # south-west (bottom-left)
            N[1][padded_mask[0:0 + col, 2:2 + row]] += N_pax_tmp[6][np.logical_and(mask, border[6])]
            # south (bottom)
            N[1][padded_mask[0:0 + col, 1:1 + row]] += N_pax_tmp[7][np.logical_and(mask, border[7])]
            # south-east (bottom-right)
            N[1][padded_mask[0:0 + col, 0:0 + row]] += N_pax_tmp[8][np.logical_and(mask, border[8])]

            # initialize goal
            N[1][self.end[0], self.end[1]] = 0.0

            self.D += N[1]

            if self.VISUALIZE:
                dsp = self.__color_map_cumulative_prob(self.D)
                dsp[dsp < 1] = self.img[dsp < 1]
                dsp = cv2.addWeighted(dsp, 0.5, self.img, 0.5, 0)
                cv2.imshow("Forecast Distribution", dsp)
                cv2.waitKey(1)

            n0_tmp, n1_tmp = N
            N = [n1_tmp, n0_tmp]

            n += 1
            if n > 300:
                break

        # output
        np.save(output_prob_filename, self.D)
        cv2.imwrite(output_image_filename, dsp)

    def __color_map(self, input_src):
        src = input_src.copy()
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)
        isInf = cv2.compare(src, -sys.float_info.max, cv2.CMP_GT)
        src[src <= -sys.float_info.max] = 0
        minVal = cv2.minMaxLoc(src)[0]
        if maxVal == minVal:
            im = np.zeros(src.shape, dtype='float')
        else:
            im = (src - minVal) / (maxVal - minVal) * 255.
        unsigned8 = im.astype(np.uint8)
        hsv = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)
        hsv[:, :, 0] = unsigned8.copy()
        hsv[:, :, 1] = isInf.copy()
        hsv[:, :, 2] = isInf.copy()
        dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return dst

    def __color_map_cumulative_prob(self, input_src):
        im = input_src.copy()
        hsv = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)

        minVal = 1e-4
        maxVal = 0.2

        im[im <= minVal] = 0.
        im = (im - minVal) / (maxVal - minVal) * 255.

        # hue
        hsv[:, :, 0] = im.astype(np.uint8).copy()
        # saturation
        im_sat = ((-im.astype(np.float64)/255.) + 1.0) * 255.0
        hsv[:, :, 1] = im_sat.astype(np.uint8).copy()
        # value
        isNonzero = cv2.compare(im, 0, cv2.CMP_GT)
        hsv[:, :, 2] = isNonzero.copy()

        dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return dst

    def __make_border_mask(self, input_img_size):

        top = np.ones(input_img_size, dtype=np.bool)
        top[0, :] = False
        bottom = np.ones(input_img_size, dtype=np.bool)
        bottom[input_img_size[0]-1, :] = False
        left = np.ones(input_img_size, dtype=np.bool)
        left[:, 0] = False
        right = np.ones(input_img_size, dtype=np.bool)
        right[:, input_img_size[1]-1] = False

        border = []
        border.append(np.logical_and(top, left))              # top-left
        border.append(top.copy())                             # top
        border.append(np.logical_and(top, right))             # top-right
        border.append(left.copy())                            # left
        border.append(np.ones(input_img_size, dtype=np.bool)) # center (not used)
        border.append(right.copy())                           # right
        border.append(np.logical_and(bottom, left))           # bottom-left
        border.append(bottom.copy())                          # bottom
        border.append(np.logical_and(bottom, right))          # bottom-right

        return border

    def compute_value_function_old(self, output_filename):
        """
        This is naive implementation ported from C++ and quite slow.
        """

        print "compute value function (quite slow) ..."

        self.reward = np.zeros(self.feature_map[0].shape, dtype='float')
        for i in range(self.w.shape[0]):
            self.reward += self.w[i] * self.feature_map[i]

        if self.VISUALIZE:
            dsp = self.__color_map(self.reward)
            cv2.imshow("Reward Function", dsp)
            cv2.waitKey(100)

        self.V = np.ones(self.feature_map[0].shape, dtype='float') * -sys.float_info.max
        V = self.V.copy()

        n = 0
        while True:
            v = self.V.copy() * 1.0
            V_padded = cv2.copyMakeBorder(v, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                          0, -sys.float_info.max)
            V_padded *= 1.0

            for col in range(V_padded.shape[0] - 2):
                for row in range(V_padded.shape[1] - 2):
                    sub = V_padded[col:col + 3, row:row + 3]
                    minVal, maxVal = cv2.minMaxLoc(sub)[0:2]

                    if maxVal == -sys.float_info.max:
                        continue

                    # softmax
                    for y in range(3):
                        for x in range(3):
                            if y == 1 and x == 1:
                                continue

                            minv = min(self.V[col, row], sub[y, x])
                            maxv = max(self.V[col, row], sub[y, x])
                            softmax = maxv + np.log(1.0 + np.exp(minv - maxv))
                            self.V[col, row] = softmax
                    self.V[col, row] += self.reward[col, row]

            self.V[self.end[0], self.end[1]] = 0.0

            # convergence criteria
            residual = cv2.absdiff(self.V, V)
            minVal, maxVal = cv2.minMaxLoc(residual)[0:2]
            V = self.V.copy()

            if maxVal < 0.9:
                break

            if self.VISUALIZE:
                dst = self.__color_map(self.V)
                dst = cv2.addWeighted(self.img, 0.5, dst, 0.5, 0)
                cv2.imshow("MaxEnt Value Function", dst)
                cv2.waitKey(1)

            n += 1
            if n > 1000:
                print "ERROR: max number of iterations"
                sys.exit(-1)

        print "converged in %d steps" % n

        # output value function
        np.save(output_filename, self.V)

    def compute_forecast_dist_old(self, output_image_filename, output_prob_filename):
        """
        This is naive implementation ported from C++ and quite slow.
        """

        print "compute forecast dist (quite slow) ..."
        self.D = np.zeros(self.V.shape, dtype='float')
        N = [self.D.copy(), self.D.copy()]
        N[0][self.start[0], self.start[1]] = 1.0

        n = 0
        while True:
            print "loop:", n
            N[1] *= 0

            for col in range(N[0].shape[0]):
                for row in range(N[0].shape[1]):
                    if col == self.end[0] and row == self.end[1]:
                        continue

                    # sys.float_info.min = 2.22507385851e-308
                    if N[0][col, row] > sys.float_info.min:

                        col_1 = N[1].shape[0] - 1
                        row_1 = N[1].shape[1] - 1

                        if col > 0 and row > 0:          # North-West
                            N[1][col - 1, row - 1] += N[0][col, row] * self.pax[0][col, row]

                        if col > 0:                      # North
                            N[1][col - 1, row - 0] += N[0][col, row] * self.pax[1][col, row]

                        if col > 0 and row < row_1:      # North-East
                            N[1][col - 1, row + 1] += N[0][col, row] * self.pax[2][col, row]

                        if row > 0:                      # West
                            N[1][col - 0, row - 1] += N[0][col, row] * self.pax[3][col, row]

                        if row < row_1:                  # East
                            N[1][col - 1, row + 1] += N[0][col, row] * self.pax[5][col, row]

                        if col < col_1 and row > 0:      # South-West
                            N[1][col + 1, row - 1] += N[0][col, row] * self.pax[6][col, row]

                        if col < col_1:                  # South
                            N[1][col + 1, row - 0] += N[0][col, row] * self.pax[7][col, row]

                        if col < col_1 and row < row_1:  # South-East
                            N[1][col + 1, row + 1] += N[0][col, row] * self.pax[8][col, row]

            N[1][self.end[0], self.end[1]] = 0.0

            self.D += N[1]

            if self.VISUALIZE:
                dsp = self.__color_map_cumulative_prob(self.D)
                dsp[dsp < 1] = self.img[dsp < 1]
                dsp = cv2.addWeighted(dsp, 0.5, self.img, 0.5, 0)
                cv2.imshow("Forecast Distribution", dsp)
                cv2.waitKey(100)

            n0_tmp, n1_tmp = N
            N = [n1_tmp, n0_tmp]

            n += 1
            if n > 500:
                break

        # output
        np.save(output_prob_filename, self.D)
        cv2.imwrite(output_image_filename, dsp)


if __name__ == '__main__':

    import time

    # input data's file names
    birdseye_image_jpg_path    = "oc_demo/walk_birdseye.jpg"
    terminal_pts_txt_path      = "oc_demo/walk_terminal_pts.txt"
    reward_weights_txt_path    = "oc_demo/walk_reward_weights.txt"
    features_npy_path          = "oc_demo/walk_feature_maps.npy"
    # output data's file names
    output_value_func_npy_path = "oc_demo/output/walk_value_function.npy"
    output_policy_npy_path     = "oc_demo/output/walk_policy.npy"
    output_prob_npy_path       = "oc_demo/output/walk_forecast_prob.npy"
    output_jpg_path            = "oc_demo/output/walk_forecast.jpg"

    model = OptimalControl()

    # load data
    model.load_terminal_pts(terminal_pts_txt_path)
    model.load_reward_weights(reward_weights_txt_path)
    model.load_features(features_npy_path)
    model.load_image(birdseye_image_jpg_path)
    model.set_named_window()

    # inference
    t1 = time.time()
    model.compute_value_function(output_value_func_npy_path)
    t2 = time.time()
    model.compute_policy(output_policy_npy_path)
    t3 = time.time()
    model.compute_forecast_dist(output_jpg_path, output_prob_npy_path)
    t4 = time.time()
    print t4 - t1, "s"

