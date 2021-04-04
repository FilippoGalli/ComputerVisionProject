import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import math
import time


def ComputeModelResolution(points, kdtree):

        resolution = 0.0
        for point in points:
            distances, indices = kdtree.query([point], 2)
            resolution += distances[0][1]

        resolution /= len(points)

        return resolution

def IsLocalMaxima(query_index, indices, saliency):
    for i in indices:
        if saliency[query_index] < saliency[i]:
            return False
        
    return True

def distance(x, y):

    distance = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)

    return distance


def gaussian_filter(tree, sigma, points, index):

    indices = tree.query_radius([points[index]], r=2*sigma)

    numerator = 0
    denominator = 0
 
    for i in indices[0]:
                        
        exp_coeff = - distance(points[index], points[i])**2 / (2 * sigma**2)
        e = math.exp(exp_coeff)
        numerator += points[i] * e
        denominator += e
       
    g = numerator / denominator

    return g

def compute_SP(pcd, sigma=None):

    tic = time.time()

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    threshold = 0.70
    min_neighbors = 5

    saliency = np.full(len(points), -1.0)
    keypoints_indices = []
    
   
    kdtree = KDTree(points)

    if sigma == None:
        resolution = ComputeModelResolution(points, kdtree)
        sigma = [resolution, 2 * resolution]
        print(resolution)

    for i in range(len(points)):
        g1 = gaussian_filter(kdtree, sigma[0], points, i)
        g2 = gaussian_filter(kdtree, sigma[1], points, i)
        saliency[i] = np.linalg.norm(np.dot(g1 - g2, normals[i]))
      

    max_saliency = np.amax(saliency)
    min_saliency = np.amin(saliency)
    mean = np.mean(saliency)


    for i in range(len(points)):
        saliency[i] = (saliency[i] - min_saliency) / (max_saliency - min_saliency)

    for i in range(len(points)):
        indices = kdtree.query_radius([points[i]], r=5*sigma[1])
        if saliency[i] >= threshold and IsLocalMaxima(i, indices[0], saliency):
            keypoints_indices.append(i)
            
    
    toc = 1000 * (time.time() - tic)
    print("SP Computation took {:.0f} [s]".format(toc/1000))
    print(f'number of keypoints found: {len(points[keypoints_indices])}')

    return [keypoints_indices, saliency]





def computeFeatureDescriptor(points, normals, saliency, kp_indices, sigma):


    keypoints = points[kp_indices]
    keypoints_normals = normals[kp_indices]
    keypoints_saliency = saliency[kp_indices]
    
    #local reference system
    origin = []
    x = []
    y = []
    z = []

    M = 3
    L = 36
    
    kdtree = KDTree(points)
    descriptor_list = []

    
    for i in range(len(keypoints)):
        
        origin.append(keypoints[i])
        z.append(keypoints_normals[i])
        x.append([1.0, 0.0, 0.0])
        y.append(np.cross(keypoints_normals[i], [1.0, 0.0, 0.0]))

        indices = kdtree.query_radius([keypoints[i]], r=sigma)
        indices = indices[0]
       
        descriptor = np.zeros((2, M, L))
        average_normal = np.zeros((M, L, 3))
        average_saliency = np.zeros((M, L, 1))

        grid_element_counter = np.zeros((M, L, 1))
        

        for j in range(len(indices)):
            v = points[indices[j]] - keypoints[i]

            v_norm = np.linalg.norm(v)
            if v_norm == 0.0:
                continue
            v_normalized = v / v_norm
          
            phi = math.acos(np.dot(v_normalized, z[i]))

            temp = v - v_norm * math.cos(phi) * z[i]

            v_xy = (temp) / (np.linalg.norm(temp))

            if np.dot(v_xy, y[i]) >= 0:
                theta = math.acos(np.dot(v_xy, x[i]))
            else:
                theta = 2 * 3.14 - math.acos(np.dot(v_xy, x[i]))
            
            p = keypoints[i] +  v_norm * v_xy

            m = int(v_norm * (M / sigma) + 0.5)
            l = int((theta * L) / (2 * 3.14) + 0.5)


            if m == M:
                m = M - 1
            if l == L:
                l = L - 1
           
            grid_element_counter[m][l] += 1 
            average_normal[m][l] += normals[indices[j]]
            average_saliency[m][l] += saliency[indices[j]]

            

        for j in range(M):
            for t in range(L):
                if grid_element_counter[j][t] != 0:
                    average_normal[j][t] /= grid_element_counter[j][t]
                    average_saliency[j][t] /= grid_element_counter[j][t]

                    delta_normals = 1.0 - abs(np.dot(average_normal[j][t], keypoints_normals[i]))
                    delta_saliency = 1.0 - average_saliency[j][t] / keypoints_saliency[i]

                    descriptor[0][j][t] = delta_normals
                    descriptor[1][j][t] = delta_saliency[0]

        
        descriptor_list.append(descriptor)

    return descriptor_list



        
def main():

    sigma = [0.4, 0.8]
    flag = True

    # Read .ply file
    input_file = "./data/bunny/reconstruction/bun_zipper.ply"
    #mesh = o3d.io.read_triangle_mesh(input_file)
    #mesh.compute_vertex_normals()

    pcd = o3d.io.read_point_cloud(input_file)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=5)

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    #pcd.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if flag:
        keypoints_indices, saliency = compute_SP(pcd)

        np.save('./data/SPkeypoints_indices', keypoints_indices)
        np.save('./data/SPsaliency', saliency)
    else:
        path_keypoints = './data/SPkeypoints_indices.npy'
        path_saliency = './data/SPsaliency.npy'
        keypoints_indices = np.load(path_keypoints)
        saliency = np.load(path_saliency)

    
    #descriptor_list = computeFeatureDescriptor(points, normals, saliency, keypoints_indices, sigma[1])

    pcd_keypoints = o3d.geometry.PointCloud()
    pcd_keypoints.points = o3d.utility.Vector3dVector(points[keypoints_indices])

    #mesh.compute_vertex_normals()
    #mesh.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_keypoints.paint_uniform_color([1.0, 0.75, 0.0])
    
    o3d.visualization.draw_geometries([pcd_keypoints, pcd])


if __name__ == '__main__':
    main()






   


























