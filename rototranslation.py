import numpy as np
import sys

def computeRototranslationParams(points1, points2):

    if len(points1) != len(points2):
        
        print('arrays have different lengths!') 
        sys.exit(0)

    x1, y1, z1 = 0.0, 0.0, 0.0
    x2, y2, z2 = 0.0, 0.0, 0.0
    N = len(points1)
    
    for i in range(len(points1)):
        x1 += points1[i][0]
        y1 += points1[i][1]
        z1 += points1[i][2]

        x2 += points2[i][0]
        y2 += points2[i][1]
        z2 += points2[i][2]

    mean1 = np.array([x1/N, y1/N, z1/N])
    mean2 = np.array([x2/N, y2/N, z2/N])
   
    q1, q2 = [], []
    H = np.zeros((3, 3))

    for i in range(len(points1)):
        q1.append(points1[i] - mean1)
        q2.append(points2[i] - mean2)
        
        H += np.atleast_2d(q1[i]).transpose() * q2[i]

    u, s, vh = np.linalg.svd(H)

    x = np.matmul(vh.transpose(), u.transpose())
    det = np.linalg.det(x)

    R = x
    T = np.atleast_2d(mean2).transpose() - np.matmul(R, np.atleast_2d(mean1).transpose())
    return [R, T]