import random 
from rototranslation import computeRototranslationParams
import numpy as np

def computeRANSAC(data):

    k = 1000
    threshold = 1.0
    error_best = 9999
    d = 5

    matching_indices = data[0]
    points1 = data[1]
    points2 = data[2]


    for i in range(k):
        
        error = 0
        n_consensus = 0
        consensus1 = []
        consensus2 = []
        sample = random.sample(range(0, len(matching_indices)), 3)
        point_list1 = []
        point_list2 = []

        for s in sample:

            match = matching_indices[s]
            point_list1.append(points1[match[0]])
            point_list2.append(points2[match[1]])

        R, T = computeRototranslationParams(point_list1, point_list2)

        if R is None or T is None:
            continue
        
        for p1 in points1:
            
            p_estimate = np.matmul(R, np.atleast_2d(p1).transpose()) + T

            for p2 in points2:
                
                residual = abs(p_estimate.transpose()[0] - p2)
            
                if all(residual <= threshold):
                    consensus1.append(p1)
                    consensus2.append(p2)
                    n_consensus += 1

        if n_consensus >= d:
            R, T = computeRototranslationParams(consensus1, consensus2)

            for y in range(len(consensus1)):
                p_estimate = np.matmul(R, np.atleast_2d(consensus1[y]).transpose()) + T
                residual = abs(p_estimate.transpose()[0] - consensus2[y])

                error += np.linalg.norm(residual)

            if error <= error_best:
                error_best = error
            
            model = [R, T]
    if error_best >= 1.0:
        print('accuracy problems')
        
    return [model, error_best]
        
        

            
                    
                        


        
            

        
