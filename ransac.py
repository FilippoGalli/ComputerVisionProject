import random 
from rototranslation import computeRototranslationParams
import numpy as np

def computeRANSAC(matching_indices, points1, points2, k=1000, minimum_consensus = 5, threshold=2.0):

    error_best = 99999 #generic big number
    consensus_best = 0
    model = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0], [0], [0]]]
    
    
    for i in range(k):
        
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
        

      
        count = 0
        for l in range(len(point_list1)):
            stima = np.matmul(R, np.atleast_2d(point_list1[l]).transpose()) + T
            residual = abs(stima.transpose()[0] - point_list2[l])

            if all(residual <= threshold):
              
                count += 1
                #print(f'punto1 {point_list1[l]} punto 2 {point_list2[l]}  stima {np.matmul(R, np.atleast_2d(point_list1[l]).transpose()) + T}')
               
        if count == 3:
            
            for m in matching_indices:
            
                p_estimate = np.matmul(R, np.atleast_2d(points1[m[0]]).transpose()) + T
                residual = abs(p_estimate.transpose()[0] - points2[m[1]])

                if all(residual <= threshold):
                
                    consensus1.append(points1[m[0]])
                    consensus2.append(points2[m[1]])
                    n_consensus += 1
                    
        
        
            #if the consensus value is greater than the minimum_consensus, estimate the model's parameters
            #using all the points that belong to the consensus
        
            if n_consensus >= minimum_consensus:
            
                error = 0
                R, T = computeRototranslationParams(consensus1, consensus2)
                if R is not None and  T is not None:
                
                    for y in range(len(consensus1)):
                        p_estimate = np.matmul(R, np.atleast_2d(consensus1[y]).transpose()) + T
                    
                        residual = abs(p_estimate.transpose()[0] - consensus2[y])
                        #print(f'consensus2 {consensus2[y]} estimate {p_estimate.transpose()[0]}  residual {residual} norm {np.linalg.norm(residual)}')
                        error += np.linalg.norm(residual)
                    
                
                    if error <= error_best and n_consensus >= consensus_best:
                        
                        consensus_best = n_consensus
                        error_best = error
                        model = [R, T]
                       

                    
                    
    
    
        
    return [model[0], model[1], error_best]
        
        

            
                    
                        


        
            

        
