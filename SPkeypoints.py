import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import math
import time


def IsLocalMaxima(query_index, indices, saliency):
    for i in indices:
        if saliency[query_index] < saliency[i]:
            return False
        
    return True

def gaussian_filter(tree, sigma, points, index):

    indices = tree.query_radius([points[index]], r=2*sigma)

    numerator = 0
    denominator = 0
 
    for i in indices[0]:
        
        exp_coeff = - (points[index] - points[i])**2 / (2 * sigma**2)
        e = np.array(list(map(lambda x : math.exp(x), exp_coeff)))
        numerator = numerator + points[i] * e
        denominator = denominator + e
       
    g = numerator / denominator

    return g

def DoG(tree, sigma, points, index):
    g1 = gaussian_filter(tree, sigma, points, index)
    g2 = gaussian_filter(tree, 2*sigma, points, index)

    return g1 - g2

def compute_SP(mesh, sigma):

    tic = time.time()


    points = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    threshold = 0.70
    min_neighbors = 5

    saliency = np.full(len(points), -1.0)
    keypoints_index = []
    saliency_keypoint = []
   
    
    kdtree = KDTree(points)

    for i in range(len(points)):
        DoG_value = DoG(kdtree, sigma, points, i)
        saliency[i] = np.linalg.norm(np.dot(DoG_value, normals[i]))
        
    max_saliency = np.amax(saliency)
    min_saliency = np.amin(saliency)
    mean = np.mean(saliency)


    for i in range(len(points)):
        saliency[i] = (saliency[i] - min_saliency) / (max_saliency - min_saliency)
        if saliency[i] >= 0.7:
            
            kp_indices = kdtree.query_radius([points[i]], 5)
            if len(kp_indices[0]) - 1  > min_neighbors and IsLocalMaxima(i, kp_indices[0], saliency):
                keypoints_index.append(i)
                saliency_keypoint.append(saliency[i])
    
    toc = 1000 * (time.time() - tic)
    print("SP Computation took {:.0f} [s]".format(toc/1000))
    print(f'number on keypoints {len(points[keypoints_index])}')

    return [points[keypoints_index], saliency_keypoint]


def main():
    sigma = 0.5

    # Read .ply file
    input_file = "./data/Armadillo.ply"
    mesh = o3d.io.read_triangle_mesh(input_file)

    keypoints, saliency = compute_SP(mesh, sigma)
    np.save('SPkeypoints', keypoints)
    np.save('SPsaliency', saliency)

    pcd_keypoints = o3d.geometry.PointCloud()
    pcd_keypoints.points = o3d.utility.Vector3dVector(keypoints)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_keypoints.paint_uniform_color([1.0, 0.75, 0.0])
    
    o3d.visualization.draw_geometries([pcd_keypoints, mesh])

    
if __name__ == '__main__':
    main()






   


























