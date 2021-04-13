import random 
from rototranslation import computeRototranslationParams
import numpy as np

def computeRANSAC(matching_indices, points1, points2, k=1000, minimum_consensus = 5, threshold=2.0):

    error_best = 99999 #generic big number
    model = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0], [0], [0]]]
    

    for i in range(k):
        
        error = 0
        n_consensus = 0
        consensus1 = []
        consensus2 = []
        point_list1 = []
        point_list2 = []

        #extract a random sample and compute the model's parameters
        sample = random.sample(range(0, len(matching_indices)), 3)

        for s in sample:

            match = matching_indices[s]
            point_list1.append(points1[match[0]])
            point_list2.append(points2[match[1]])

        R, T = computeRototranslationParams(point_list1, point_list2)

        if R is None or T is None:
            continue
        

        #determine the value of the consesus that is compatible with the model
        for p1 in points1:
            
            p_estimate = np.matmul(R, np.atleast_2d(p1).transpose()) + T

            for p2 in points2:
                
                residual = abs(p_estimate.transpose()[0] - p2)
            
                if all(residual <= threshold):
                    consensus1.append(p1)
                    consensus2.append(p2)
                    n_consensus += 1

        #if the consensus value is greater than the minimum_consensus, estimate the model's parameters
        #using all the points that belong to the consensus

        if n_consensus >= minimum_consensus:
            R, T = computeRototranslationParams(consensus1, consensus2)
            if R is not None or T is not None:
                
                for y in range(len(consensus1)):
                    p_estimate = np.matmul(R, np.atleast_2d(consensus1[y]).transpose()) + T
                    residual = abs(p_estimate.transpose()[0] - consensus2[y])

                    error += np.linalg.norm(residual)
                
                if error <= error_best:
                    error_best = error
                    model = [R, T]
    
    
        
    return [model[0], model[1], error_best]
        
        

            
                    
                        


        
            

        
