import open3d as o3d
from sklearn.neighbors import KDTree
import numpy as np
import math


def gaussianFilter(points, sigma, index, kdtree=None):
    
    if kdtree == None:
        kdtree = KDTree(points)
    
    indices, distances = kdtree.query_radius([points[index]], r=2*sigma, return_distance=True)

    #refactor
    indices = indices[0]
    distances = distances[0]

    numerator = 0
    denominator = 0
 
    for i in range(len(indices)):
        exp_coeff = -distances[i]**2 / (2 * sigma**2)
        e = math.exp(exp_coeff)

        numerator += points[indices[i]] * e
        denominator += e
    
    g = numerator / denominator

    return g

def main():

    # Read .ply file
    input_file = "./data/Armadillo.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(5)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    #variables
    sigma = [0.3, 0.6, 1.0]
    g1 = []
    g2 = []
    g3 = []
    saliency = []

    #pcd2= pcd.uniform_down_sample(2)

    
    #compute gaussian filter with different sigma and points resolution
    for i in range(len(points)):
        g1 = gaussianFilter(points, sigma[0], i)  #use all pcd points
        g2 = gaussianFilter(points, sigma[0], i)
        saliency.append(np.linalg.norm(np.dot(normals[i], g1 - g2)))

    

if __name__ == '__main__':
    main()

