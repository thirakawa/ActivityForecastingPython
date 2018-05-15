# -*- coding: utf-8 -*-

import sys
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


class InverseOptimalControl(object):
    """Inverse Optimal Control (Inverse Reinforcement Learning)"""
    def __init__(self, visualize=True, verbose=True):

        self.basenames        = []                       # file basenames
        self.trajectory_gt    = []                       # ground truth trajectory
        self.trajectory_obs   = []                       # observed tracker output
        self.feature_map      = []                       # (physical) feature maps
        self.images           = []                       # (physical) feature maps
        
        self.pax              = []                       # policy [_nd _na]
        self.reward           = []                       # Reward Function
        self.value            = []                       # Soft Value Function
        self.w                = []                       # reward parameters
        self.f_empirical      = []                       # empirical feature count
        self.f_expected       = []                       # expected feature count
        self.f_gradient       = []                       # gradient
        
        self.w_best           = []                       # reward parameters
        self.f_gradient_best  = []                       # gradient
        
        self.start            = []                       # terminal states
        self.end              = []                       # start states
        self.n_feature        = None                     # number of feature types
        self.n_data           = None                     # number of training data
        self.n_action         = 9                        # number of actions [3x3]
        self.size             = None                     # current state space size
        self.loglikelihood    = 0                        #
        self.minloglikelihood = -sys.float_info.max      #
        self.lam              = 0.01                     # exp-gradient descent step size
        
        self.error            = 0                        # bad parameters
        self.converged        = False
        
        self.VISUALIZE        = visualize                # visualization flag
        self.VERBOSE          = verbose                  # print out intermediate status
        self.DELTA            = 0.01                     # minimum improvement of log-likelihood

    # data loading ============================================
    def load_basenames(self, input_filename):
        print "\nload basenames..."
        with open(input_filename) as f:
            self.basenames = map(self.__erase_new_line, f.readlines())

        # set the number of data
        self.n_data = len(self.basenames)
        if self.VERBOSE: print "    number of basenames loaded: %d" % self.n_data

    def load_trajectories(self, input_file_prefix):
        print "\nload trajectories..."
        for base in self.basenames:
            loaded_traj = np.loadtxt(input_file_prefix + base + "_tracker_output.txt", dtype='int')
            self.trajectory_gt.append(np.c_[loaded_traj[:,2], loaded_traj[:,1]])
            self.trajectory_obs.append(np.c_[loaded_traj[:,4], loaded_traj[:,3]])

            # set start and end coordinate from ground-truth [y, x]
            self.start.append(np.c_[loaded_traj[:, 2], loaded_traj[:, 1]][0, 0:2])
            self.end.append(np.c_[loaded_traj[:, 2], loaded_traj[:, 1]][-1, 0:2])

            # print trajectory length
            if self.VERBOSE: print "    %s: trajectory length: %d" % (base, loaded_traj.shape[0])
        if self.VERBOSE: print "    number of trajectories loaded: %d" % len(self.trajectory_gt)

    def load_feature_maps(self, input_file_prefix):
        print "\nload feature maps..."
        for base in self.basenames:
            loaded_feature_map = np.load(input_file_prefix + base + "_feature_maps.npy")
            self.feature_map.append(loaded_feature_map)
            self.n_feature = loaded_feature_map.shape[0] - 3
            self.size = loaded_feature_map[0].shape
            if self.VERBOSE: print "    %s: number of features loaded is %d" % (base, self.n_feature)

    def load_images(self, input_file_prefix):
        print "\nload images..."
        for base in self.basenames:
            loaded_img = cv2.imread(input_file_prefix + base + "_topdown.jpg")
            resized_img = cv2.resize(loaded_img, (self.size[1], self.size[0]))
            self.images.append(resized_img)
        if self.VERBOSE: print "    number of images loaded: %d" % len(self.images)

    # initialize ==============================================
    def initialize(self):
        print "\ninitializing parameters..."

        # weight
        self.w = np.ones(self.n_feature, dtype='float') * 0.5

        # reward
        self.reward = [np.zeros(self.size, dtype='float') for i in range(self.n_data)]

        # soft value function
        self.value = [np.zeros(self.size, dtype='float') for i in range(self.n_data)]

        # number of actions
        self.n_action = 9

        # policy
        self.pax = [np.zeros([self.n_action, self.size[0], self.size[1]], dtype='float') for i in range(self.n_data)]

        # empirical feature count, expected feature count, gradient
        self.f_empirical = np.zeros(self.n_feature, dtype='float')
        self.f_expected = np.zeros(self.n_feature, dtype='float')
        self.f_gradient = np.zeros(self.n_feature, dtype='float')

        self.minloglikelihood = -sys.float_info.max

        if self.VISUALIZE:
            cv2.namedWindow("Reward Function")
            cv2.namedWindow("MaxEnt Value Function")
            cv2.namedWindow("Forecast Distribution")
            cv2.moveWindow("Reward Function", 0, 0)
            cv2.moveWindow("MaxEnt Value Function", self.size[1], 0)
            cv2.moveWindow("Forecast Distribution", self.size[1] * 2, 0)

    # compute empirical statistics ============================
    def compute_empirical_statistics(self):
        print "\ncompute empirical statistics..."

        for i, base in enumerate(self.basenames):
            if self.VERBOSE: print "    add feature counts for %s" % base

            for point in self.trajectory_gt[i]:
                self.__accumulate_empirical_feature_counts(i, point)

        self.f_empirical = self.f_empirical / self.n_data
        if self.VERBOSE:
            print "    mean empirical feature count:\n", np.vectorize("%.3f".__mod__)(self.f_empirical)

    def __accumulate_empirical_feature_counts(self, data_i, pt):
        for i in range(self.n_feature):
            self.f_empirical[i] += self.feature_map[data_i][i, pt[0], pt[1]]

    # backward pass ==========================================
    def backward_pass(self):
        print "\nbackward pass..."
        self.error = 0
        self.loglikelihood = 0

        for d in range(self.n_data):
            print "   ", self.basenames[d], "===================="
            self.__compute_reward_function(d)
            self.value[d] = self.__compute_soft_value_function(self.reward[d], self.end[d], self.images[d])
            self.pax[d] = self.__compute_policy(self.pax[d], self.value[d], self.n_action)
            self.loglikelihood += self._compute_trajectory_likelihood(self.pax[d], self.trajectory_gt[d])

            if self.loglikelihood <= -sys.float_info.max: break

        print "\n    loglikelihood sum:", self.loglikelihood

    def __compute_reward_function(self, data_i):
        if self.error: return None
        if self.VERBOSE: print "    compute reward function"

        self.reward[data_i] *= 0
        for f in range(self.n_feature):
            self.reward[data_i] += self.w[f] * self.feature_map[data_i][f]

        # plt.imshow(self.reward[data_i], cmap='jet')
        # plt.colorbar()
        # plt.show()

        if self.VISUALIZE:
            colorized_reward_map = self.__color_map(self.reward[data_i])
            cv2.imshow("Reward Function", colorized_reward_map)
            cv2.waitKey(1)

    def __compute_soft_value_function(self, input_reward, input_end, input_image):
        if self.error: return None
        if self.VERBOSE: print "    compute soft value function (modified; faster)"

        V = [np.ones(input_reward.shape, dtype='float') * -sys.float_info.max,
             np.ones(input_reward.shape, dtype='float') * -sys.float_info.max]

        V[0][input_end[0], input_end[1]] = 0.0

        n = 0
        while True:
            v = V[0].copy() * 1.0
            V_padded = cv2.copyMakeBorder(v, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                          value=-sys.float_info.max)
            V_padded *= 1.0

            sub_array = np.zeros([self.n_action, v.shape[0], v.shape[1]], dtype='float')

            sub_arr_index = 0
            for y in range(3):
                for x in range(3):
                    sub_array[sub_arr_index, :, :] = V_padded[y:y+v.shape[0], x:x+v.shape[1]].copy()
                    sub_arr_index += 1

            is_neg_inf = sub_array == -sys.float_info.max
            neg_inf_mat = np.logical_not(np.prod(is_neg_inf, axis=0))

            for i, sub_elem in enumerate(sub_array):
                if i == 4:
                    continue
                minv = np.minimum(V[0], sub_elem)
                maxv = np.maximum(V[0], sub_elem)
                softmax = maxv + np.log(1.0 + np.exp(minv - maxv))
                V[0][neg_inf_mat] = softmax[neg_inf_mat]
            # print "max vals", np.max(V[0][neg_inf_mat])
            V[0][neg_inf_mat] += input_reward[neg_inf_mat]

            # elements of V[0] should be lower than 1
            if np.sum(V[0][neg_inf_mat]  > 0) > 0:
                self.error = 1
                return None

            # init goal
            V[0][input_end[0], input_end[1]] = 0.0

            # convergence criteria
            residual = cv2.absdiff(V[0], V[1])
            minVal, maxVal = cv2.minMaxLoc(residual)[0:2]
            V[1] = V[0].copy()

            if maxVal < 0.9:
                break

            if self.VISUALIZE:
                dst = self.__color_map(V[0])
                dst = cv2.addWeighted(input_image, 0.5, dst, 0.5, 0)
                cv2.imshow("MaxEnt Value Function", dst)
                cv2.waitKey(1)

            n += 1
            if n > 1000:
                print "ERROR: max number of iterations"
                self.error = 1
                return None

        return V[0]

    def __compute_policy(self, input_pax, input_value, input_n_action):
        if self.error: return None
        if self.VERBOSE: print "    compute policy..."

        V_padded = cv2.copyMakeBorder(input_value, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=-np.inf)

        for col in range(V_padded.shape[0] - 2):
            for row in range(V_padded.shape[1] - 2):
                sub = V_padded[col:col+3, row:row+3].copy()
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sub)
                p = sub - maxVal  # log scaling
                p = np.exp(p)     # Z(x,a) - probability space
                p[1,1] = 0        # zero out center
                su = np.sum(p)    # sum (denominator)
                if su > 0:        # normalize (compute policy(x|a))
                    p /= su
                else:             # uniform distribution
                    p = np.ones(p.shape, dtype='float') * (1.0 / (input_n_action - 1.0))
                p = p.flatten()
                # update policy
                for a in range(input_n_action):
                    input_pax[a][col, row] = p[a]

        return input_pax

    def _compute_trajectory_likelihood(self, input_pax, input_traj_gt):
        if self.VERBOSE: print "    compute trajectory likelihood..."

        trans_index = np.array([[0,1,2],
                                [3,-1,5],
                                [6,7,8]], dtype='int')

        ll = 0.

        for t in range(input_traj_gt.shape[0]-1):
            dx = input_traj_gt[t+1, 1] - input_traj_gt[t, 1]
            dy = input_traj_gt[t+1, 0] - input_traj_gt[t, 0]

            a = trans_index[dy+1, dx+1]

            if a < 0:
                print "ERROR: invalid action %d(%d, %d)" % (t, dx, dy)
                print "preprocess trajectory data property"
                sys.exit(-1)

            val = np.log(input_pax[a][input_traj_gt[t, 0], input_traj_gt[t, 1]])

            if val < -sys.float_info.max:
                ll = -sys.float_info.max
                break

            ll += val

        if self.VERBOSE: print "      loglikelihood:", ll
        return ll

    # forward pass ==========================================
    def forward_pass(self):
        print "\nforeward pass..."

        if self.error or self.loglikelihood < self.minloglikelihood:
            print "    skip..."
            return None

        # initialize
        self.f_expected = np.zeros(self.n_feature, dtype='float')

        for d in range(self.n_data):
            print "   ", self.basenames[d], "===================="
            D = self.__compute_state_visitation_dist(self.pax[d], self.start[d], self.end[d], self.images[d])
            self.__accumulate_expected_feature_counts(D, self.feature_map[d])

        print "    mean expected feature count:", self.f_expected

    def __compute_state_visitation_dist(self, input_pax, input_start, input_end, input_image):
        if self.VERBOSE:
            print "    compute state visitation distribution (modified; faster)"

        N = [np.zeros(self.size, dtype='float'),
             np.zeros(self.size, dtype='float')]
        N[0][input_start[0], input_start[1]] = 1.0
        col = N[0].shape[0]
        row = N[0].shape[1]

        D = np.zeros(self.size, dtype='float')
        D += N[0]

        border = self.__make_border_mask(self.size)

        n = 0
        while True:
            N[1] *= 0.0

            mask = np.zeros(N[0].shape, dtype=np.bool)
            mask[N[0] > sys.float_info.min] = True
            mask[input_end[0], input_end[1]] = False
            padded_mask = np.lib.pad(mask, 1, 'constant', constant_values=False)

            N_pax_tmp = []
            for i in range(9):
                N_pax_tmp.append(N[0] * input_pax[i])

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
            N[1][input_end[0], input_end[1]] = 0.0

            n0_tmp, n1_tmp = N
            N = [n1_tmp, n0_tmp]

            D += N[1]

            if self.VISUALIZE:
                dsp = self.__color_map_cumulative_prob(D)
                dsp[dsp < 1] = input_image[dsp < 1]
                dsp = cv2.addWeighted(dsp, 0.5, input_image, 0.5, 0)
                cv2.imshow("Forecast Distribution", dsp)
                cv2.waitKey(1)

            n += 1
            if n > 300:
                break

        return D

    def __accumulate_expected_feature_counts(self, input_D, input_feature_maps):
        if self.VERBOSE: print "    accumulate expected feature counts"
        for f in range(self.n_feature):
            F = input_D * input_feature_maps[f,:,:]
            self.f_expected[f] += np.sum(F) / float(self.n_data)

    # gradient update ========================================
    def gradient_update(self):
        print "\ngradient update..."

        if self.error:
            print "    ERROR: Increase step size"
            for f in range(self.n_feature):
                self.w[f] *= 2.0

        # compute likelihood improvement
        improvement = self.loglikelihood - self.minloglikelihood

        if improvement > self.DELTA:
            self.minloglikelihood = self.loglikelihood
        elif improvement < self.DELTA and improvement >= 0:
            improvement = 0
        elif improvement > -self.DELTA and improvement <= 0:
            improvement = 0

        if self.VERBOSE:
            print "    improved by:", improvement

        # update parameters (standard line search)
        if improvement < 0:
            print "    ===> NO IMPROVEMENT: decrease step size and redo"
            self.lam = self.lam * 0.5
            for f in range(self.n_feature):
                self.w[f] = self.w_best[f] * math.exp( self.lam * self.f_gradient[f])
        elif improvement > 0:
            print "    ===> IMPROVEMENT: increase step size"
            self.w_best = self.w.copy()
            self.lam = self.lam * 2.0
            for f in range(self.n_feature):
                self.f_gradient[f] = self.f_empirical[f] - self.f_expected[f]
            for f in range(self.n_feature):
                self.w[f] = self.w_best[f] * math.exp(self.lam * self.f_gradient[f])
        elif improvement == 0:
            print "    CONVERGED"
            self.converged = True

        if self.VERBOSE:
            print "    lambda:", self.lam
            print "    f_empirical:", self.f_empirical
            print "    f_expected:", self.f_expected

    # save parameters ========================================
    def save_parameters(self, output_filename):
        base, ext = os.path.splitext(output_filename)
        if ext == ".npy":
            np.save(output_filename, self.w_best)
        elif ext == ".txt":
            np.savetxt(output_filename, self.w_best)
        else:
            print "WARNING: file extension should be txt or npy."
            print "         parameters are saved as", base + ".txt"
            np.savetxt(base + ".txt", self.w_best)

    # other private functions ================================
    def __erase_new_line(self, input_string):
        return input_string.strip()

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

    def __compute_soft_value_function_old(self, input_reward, input_end, input_image):
        """
        This is naive implementation ported from C++ and quite slow.
        """

        if self.error: return None
        if self.VERBOSE: print "    compute soft value function (quite slow)"

        V = [np.ones(input_reward.shape, dtype='float') * -sys.float_info.max,
             np.ones(input_reward.shape, dtype='float') * -sys.float_info.max]

        n = 0
        while True:
            v = V[0].copy() * 1.0
            V_padded = cv2.copyMakeBorder(v, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=-sys.float_info.max)
            V_padded *= 1.0

            # beginning of nested loop ======
            for col in range(V_padded.shape[0] - 2):
                for row in range(V_padded.shape[1] - 2):
                    sub = V_padded[col:col+3, row:row+3].copy()
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sub)

                    if maxVal == -sys.float_info.max:
                        continue

                    for y in range(3):
                        for x in range(3):
                            if y == 1 and x == 1:
                                continue

                            minv = min(V[0][col, row], sub[y, x])
                            maxv = max(V[0][col, row], sub[y, x])
                            softmax = maxv + math.log(1.0 + math.exp(minv - maxv))
                            V[0][col, row] = softmax

                    V[0][col, row] += input_reward[col, row]

                    if V[0][col, row] > 0:
                        self.error = 1
                        return None
            # end of nested loop ============

            # init goal
            V[0][input_end[0], input_end[1]] = 0.0

            # convergence criteria
            residual = cv2.absdiff(V[0], V[1])
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(residual)
            V[1] = V[0].copy()

            if maxVal < 0.9:
                break

            if self.VISUALIZE:
                cmap = self.__color_map(V[0])
                cv2.imshow("MaxEnt Value Function", cmap)
                cv2.waitKey(1)

            n += 1
            if n > 1000:
                print "ERROR: max number of iterations."
                self.error = 1
                return None

        return V[0]

    def __compute_state_visitation_dist_old(self, input_pax, input_start, input_end, input_image):
        """
        This is naive implementation ported from C++ and quite slow.
        """

        if self.VERBOSE: print "    compute state visitation distribution (quite slow)"

        N = [np.zeros(self.size, dtype='float'),
             np.zeros(self.size, dtype='float')]
        N[0][input_start[0], input_start[1]] = 1.0

        D = np.zeros(self.size, dtype='float')
        D += N[0]

        n = 0
        while True:
            N[1] *= 0.0
            for col in range(N[0].shape[0]):
                for row in range(N[0].shape[1]):
                    if col == input_end[0] and row == input_end[1]:
                        continue

                    # sys.float_info.min = 2.22507385851e-308
                    if N[0][col, row] > sys.float_info.min:
                        col_1 = N[1].shape[0] - 1
                        row_1 = N[1].shape[1] - 1

                        if col > 0 and row > 0:          # North-West
                            N[1][col - 1, row - 1] += N[0][col, row] * input_pax[0][col, row]

                        if col > 0:                      # North
                            N[1][col - 1, row - 0] += N[0][col, row] * input_pax[1][col, row]

                        if col > 0 and row < row_1:      # North-East
                            N[1][col - 1, row + 1] += N[0][col, row] * input_pax[2][col, row]

                        if row > 0:                      # West
                            N[1][col - 0, row - 1] += N[0][col, row] * input_pax[3][col, row]

                        if row < row_1:                  # East
                            N[1][col - 0, row + 1] += N[0][col, row] * input_pax[5][col, row]

                        if col < col_1 and row > 0:      # South-West
                            N[1][col + 1, row - 1] += N[0][col, row] * input_pax[6][col, row]

                        if col < col_1:                  # South
                            N[1][col + 1, row - 0] += N[0][col, row] * input_pax[7][col, row]

                        if col < col_1 and row < row_1:  # South-East
                            N[1][col + 1, row + 1] += N[0][col, row] * input_pax[8][col, row]

            N[1][input_end[0], input_end[1]] = 0.0

            n0_tmp, n1_tmp = N
            N = [n1_tmp, n0_tmp]

            D += N[1]

            if self.VISUALIZE:
                dsp = self.__color_map_cumulative_prob(D)
                dsp[dsp < 1] = input_image[dsp < 1]
                dsp = cv2.addWeighted(dsp, 0.5, input_image, 0.5, 0)
                cv2.imshow("Forecast Distribution", dsp)
                cv2.waitKey(1)

            n += 1
            if n > 300:
                break
        return D


if __name__ == '__main__':

    import time

    basenames_txt_path        = "ioc_demo/walk_basenames.txt"
    demontraj_txt_path_prefix = "ioc_demo/walk_traj/"
    feat_maps_path_prefix     = "ioc_demo/walk_feat/"
    rect_imag_jpg_path_prefix = "ioc_demo/walk_imag/"
    output_params_path        = "ioc_demo/walk_output/walk_reward_params.txt"

    model = InverseOptimalControl(visualize=True, verbose=True)
    model.load_basenames(basenames_txt_path)
    model.load_trajectories(demontraj_txt_path_prefix)
    model.load_feature_maps(feat_maps_path_prefix)
    model.load_images(rect_imag_jpg_path_prefix)

    model.initialize()

    model.compute_empirical_statistics()

    niter = 0
    while not model.converged:
        start = time.time()
        model.backward_pass()
        model.forward_pass()
        model.gradient_update()
        model.save_parameters("ioc_demo/walk_output/walk_reward_params_%03d.txt" % niter)
        niter += 1
        end = time.time()
        print end - start, "[s]"
