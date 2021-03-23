import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import math
import time


def gaussian_filter(tree, sigma, points, index):

    indices = tree.query_radius(points[index:index+1], r=2*sigma)

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

    points = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    threshold = 0.75
   
    saliency_list = []
    tree = KDTree(points)

    for i in range(len(points)):
        DoG_value = DoG(tree, sigma, points, i)
        saliency_list.append(np.linalg.norm(np.dot(DoG_value, normals[i])))
    
    min_saliency = np.amin(saliency_list)
    max_saliency = np.amax(saliency_list)

    keypoints = []
    for i in range(len(saliency_list)):
        saliency_list[i] = (saliency_list[i] - min_saliency) / (max_saliency - min_saliency)

        if saliency_list[i] >= threshold:
            keypoints.append(points[i])

    return keypoints



def main():
    sigma = 0.5

    # Read .ply file
    input_file = "Armadillo.ply"
    mesh = o3d.io.read_triangle_mesh(input_file)

    tic = time.time()

    keypoints = compute_SP(mesh, sigma)
    print(f'number on keypoints {len(keypoints)}')
    pcd_keypoints = o3d.geometry.PointCloud()
    pcd_keypoints.points = o3d.utility.Vector3dVector(keypoints)
    
    toc = 1000 * (time.time() - tic)
    print("SP Computation took {:.0f} [ms]".format(toc))

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    #keypoints.paint_uniform_color([1.0, 0.75, 0.0])
    o3d.visualization.draw_geometries([pcd_keypoints, mesh])

    
if __name__ == '__main__':
    main()






   


























