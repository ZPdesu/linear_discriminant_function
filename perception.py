# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import *
import math

def read_data():
    w1 = np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3],
                 [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]])
    w2 = np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0],
                [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]])
    w3 = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2],
                 [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]])
    w4 = np.array([[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7],
                 [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])

    return w1, w2, w3, w4


def batch_perception(w1, w2):
    n1, d1 = np.shape(w1)
    n2, d2 = np.shape(w2)
    x1 = np.column_stack((w1, np.ones(n1)))
    x2 = np.column_stack((w2, np.ones(n2)))
    x2 = -x2
    x = np.row_stack((x1, x2))
    n, d = np.shape(x)
    max_iter = 1000
    iteration = 0
    theta = 0
    eta = 0.1
    a = np.zeros(3)
    for i in range(max_iter):
        Y = []
        for j in range(n):
            if np.dot(a, x[j]) <= 0:
                Y.append(x[j])
        a += eta * np.sum(Y, axis=0)
        iteration += 1
        if norm(np.mat(eta * np.sum(Y, axis=0)), 2) <= theta:
            break
    return a, iteration


def ho_kashyap(w1, w2):
    n1, d1 = np.shape(w1)
    n2, d2 = np.shape(w2)
    x1 = np.column_stack((w1, np.ones(n1)))
    x2 = np.column_stack((w2, np.ones(n2)))
    x2 = -x2
    x = np.row_stack((x1, x2))
    n, d = np.shape(x)
    max_iter = 1000
    iteration = 0
    eta = 1
    b = np.ones(n) * 0.01
    b_min = 0.0005
    a = np.dot(np.linalg.pinv(x), b)
    #a = np.zeros(d)
    for i in range(max_iter):
        e = np.dot(x, a) - b
        e_plus = 0.5 * (e + abs(e))
        b += 2 * eta * e_plus
        a = np.dot(np.linalg.pinv(x),b)
        iteration += 1
        #print e
        if norm(np.mat(e), 2) <= b_min:
            return a, b, iteration
    print 'No solution found'
    return a, b, iteration


def batch_rwm(w1, w2, margin=0.1):
    n1, d1 = np.shape(w1)
    n2, d2 = np.shape(w2)
    x1 = np.column_stack((w1, np.ones(n1)))
    x2 = np.column_stack((w2, np.ones(n2)))
    x2 = -x2
    x = np.row_stack((x1, x2))
    n, d = np.shape(x)
    max_iter = 100
    iteration = 0
    eta = 0.1
    b = np.ones(n) * margin
    a = np.zeros(d)
    J = []
    for i in range(max_iter):
        Y = []
        J_temp = 0
        sigma = np.zeros(d)
        for j in range(n):
            if np.dot(a, x[j]) <= b[j]:
                Y.append(x[j])
                temp = (np.dot(a, x[j]) - b[j]) / math.pow(norm(np.mat(x[j]), 2), 2) * x[j]
                sigma += temp
                temp_J = 0.5 * math.pow((np.dot(a, x[j]) - b[j]), 2) / math.pow(norm(np.mat(x[j]), 2), 2)
                J_temp += temp_J
        a -= eta * sigma
        iteration += 1
        J.append(J_temp)
        if Y == []:
            return a, b, iteration, J
    print 'No solution found'
    return a, b, iteration, J


def single_sample_rwm(w1, w2, margin=0.1):
    n1, d1 = np.shape(w1)
    n2, d2 = np.shape(w2)
    x1 = np.column_stack((w1, np.ones(n1)))
    x2 = np.column_stack((w2, np.ones(n2)))
    x2 = -x2
    x = np.row_stack((x1, x2))
    n, d = np.shape(x)
    max_iter = 200
    iteration = 0
    eta = 0.02
    b = np.ones(n) * margin
    a = np.zeros(d)
    J = []
    for i in range(max_iter):
        Y = []
        J_temp = 0
        for j in range(n):
            if np.dot(a, x[j]) <= b[j]:
                Y.append(x[j])
                temp = (np.dot(a, x[j]) - b[j]) / math.pow(norm(np.mat(x[j]), 2), 2) * x[j]
                a -= eta * temp
                iteration += 1
                for k in range(n):
                    temp_J = 0.5 * math.pow((np.dot(a, x[k]) - b[k]), 2) / math.pow(norm(np.mat(x[k]), 2), 2)
                    J_temp += temp_J
                J.append(J_temp)
        if Y == []:
            return a, b, iteration, J
    print 'No solution found'
    return a, b, iteration, J


def cal_one(w1, w2, w3, w4):
    a1, iteration1 = batch_perception(w1, w2)
    a2, iteration2 = batch_perception(w2, w3)

    print a1, iteration1
    print a2, iteration2
    plt.scatter(w1[:, 0], w1[:, 1])
    plt.scatter(w2[:, 0], w2[:, 1])
    x1 = range(-5, 10, 1)
    y1 = (- a1[0] * np.array(x1) - a1[2]) / a1[1]
    plt.plot(x1, y1, color='red', label='line')
    plt.title('1(a)')
    plt.legend()
    plt.show()

    plt.scatter(w2[:, 0], w2[:, 1])
    plt.scatter(w3[:, 0], w3[:, 1])
    x2 = range(-5, 10, 1)
    y2 = (- a2[0] * np.array(x2) - a2[2]) / a2[1]
    plt.plot(x2, y2, color='red', label='line')
    plt.title('1(b)')
    plt.legend()
    plt.show()


def cal_two(w1, w2, w3, w4):
    a1, b1, iteration1 = ho_kashyap(w1, w3)
    print a1, b1, iteration1
    a2, b2, iteration2 = ho_kashyap(w2, w4)
    print a2, b2, iteration2


def cal_three(w1, w2, w3, w4):
    a1, b1, iteration1, J1 = batch_rwm(w1, w3, 0.1)
    print a1, b1, iteration1
    #a1, b1, iteration1, J1 = batch_rwm(w1, w3, 0.5)
    #print a1, b1, iteration1
    x = range(iteration1)
    plt.plot(x, J1)
    plt.show()


def cal_four(w1, w2, w3, w4):
    a1, b1, iteration1, J1 = single_sample_rwm(w1, w3, 0.5)
    print a1, b1, iteration1
    x = range(iteration1)
    plt.plot(x, J1)
    plt.show()


# Test call
if __name__ == '__main__':
    w1, w2, w3, w4 = read_data()
    #cal_one(w1, w2, w3, w4)
    #cal_two(w1, w2, w3, w4)
    cal_three(w1, w2, w3, w4)
    #cal_four(w1, w2, w3, w4)
