from numpy import *
from scipy.linalg import *
w1 = array([[0.1, 1.1, 1], [6, 7, 1], [-3.5, -4.1, 1], [2.0, 2.7, 1], [4.1, 2.8, 1], [3.1, 5.0, 1], [-0.8, -1.3, 1],
                 [0.9, 1.2, 1], [5.0, 6.4, 1], [3.9, 4.0, 1]])



a = array([5.0,6,5,7])
b = range(4)
ttemp = mat(a).copy()

a[2] = 2
print ttemp
ttemp[0, 3] =1
print a
print type(ttemp)
print type(norm(ttemp, 2))
print type(b[1])
print b
b = b / norm(ttemp, 2)
print b
