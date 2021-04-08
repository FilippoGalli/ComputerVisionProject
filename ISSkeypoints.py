from sklearn.neighbors import KDTree
import open3d as o3d
import numpy as np
import time
import math


def computeISS(pcd, salient_radius=0, non_max_radius=0, gamma_21=0.975 , gamma_32=0.975 , min_neighbors=5):

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
    
    
    points = np.asarray(pcd.points)
    saliency = np.full(len(points), -1.0)
    keypoints_indices = []
  
    
    kdtree = KDTree(points)

    if (salient_radius == 0.0 or non_max_radius == 0.0):
        resolution = ComputeModelResolution(points, kdtree)
        salient_radius = 6 * resolution
        non_max_radius = 4 * resolution
        print(f'salient_radius= {salient_radius} non_max_radius= {non_max_radius}')
    
    tic = time.time()
    for i in range(len(points)):

        indices = kdtree.query_radius([points[i]], salient_radius)

        if len(indices[0]) - 1  < min_neighbors:
            continue
        
        support = points[indices[0]]
        support_mean = np.mean(support, axis=0)

        
        cov = np.zeros([3,3])
        for vector in support:
            vector = np.array(vector) - support_mean
            cov = cov + vector * np.atleast_2d(vector).transpose()
        cov = cov / len(support)  
        if  np.array_equal(cov, np.zeros([3, 3])):
            continue
        
        
        eigVal, eigVect = np.linalg.eig(cov)
        idx = eigVal.argsort()[::-1]   
        eigVal = eigVal[idx]
        eigVect = eigVect[:,idx]
        
        e1c = eigVal[0]
        e2c = eigVal[1]
        e3c = eigVal[2]
        

        if e2c / e1c < gamma_21 and e3c / e2c < gamma_32:
            saliency[i] = e3c
    
    for i in range(len(points)):
        if saliency[i] > 0.0:
            
            kp_indices = kdtree.query_radius([points[i]], non_max_radius)
            if len(kp_indices[0]) - 1  > min_neighbors and IsLocalMaxima(i, kp_indices[0], saliency):
                keypoints_indices.append(i)
              

    toc = 1000 * (time.time() - tic)

    print("ISS Computation took {:.0f} [s]".format(toc/1000))
    print(f'number of keypoints found: {len(points[keypoints_indices])}')

    return [keypoints_indices, saliency,  salient_radius, non_max_radius]
                  
 
 
 
def computeFeatureDescriptor(points, normals, saliency, kp_indices, radius):


    max_saliency = max(saliency)
    min_saliency = min(saliency)

    for i in range(len(saliency)):
        saliency[i] = (saliency[i] - min_saliency) / (max_saliency - min_saliency)



    keypoints = points[kp_indices]
    keypoints_normals = normals[kp_indices]
    keypoints_saliency = saliency[kp_indices]
    
    

    #local reference system
  
    x = []
    y = []
    z = []

    M = 3
    L = 36
    
    kdtree = KDTree(points)
    descriptor_list = []

    
    
    for i in range(len(keypoints)):
    
    
       
        z.append(keypoints_normals[i])

        x_first_comp = 1.0 - keypoints[i][0]
        x_second_comp = 1.0 - keypoints[i][1]
        formula = ((-z[i][0] * x_first_comp) + (-z[i][1] * x_second_comp) + (z[i][2] * keypoints[i][2])) / z[i][2]
        x_third_comp = formula - keypoints[i][2]
        x_norm = np.linalg.norm([x_first_comp, x_second_comp, x_third_comp])
        x.append([x_first_comp, x_second_comp, x_third_comp] / x_norm)
        
        y.append(np.cross(z[i], x[i]))
        

        indices, distances = kdtree.query_radius([keypoints[i]], r=radius, return_distance=True)
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

        
            temp = v - (v_norm * math.cos(phi) * z[i])

            v_xy = temp / (np.linalg.norm(temp))

            if np.dot(v_xy, y[i]) >= 0:
                theta = math.acos(np.dot(v_xy, x[i]))
            else:
                theta = (2 * math.pi) - math.acos(np.dot(v_xy, x[i]))
            
            m = int((v_norm * (M / radius)) + 0.5)
            l = int((theta * L) / (2 * math.pi) + 0.5)

            m -= 1
            l -= 1

            if m == -1:
                m = 0
            if l == -1:
                l = 0
            
            grid_element_counter[m][l] += 1 
            average_normal[m][l] += normals[indices[j]]
            average_saliency[m][l] += saliency[indices[j]]

            
        for s in range(M):
            for t in range(L):
                if grid_element_counter[s][t] != 0:

                    average_normal[s][t] /= grid_element_counter[s][t]
                    average_saliency[s][t] /= grid_element_counter[s][t]
                    delta_normals = 1.0 - abs(np.dot(average_normal[s][t], keypoints_normals[i]))
                    delta_saliency = 1.0 - (average_saliency[s][t] / keypoints_saliency[i])

                    descriptor[0][s][t] = delta_normals
                    descriptor[1][s][t] = delta_saliency[0]

      
        descriptor_list.append(descriptor)

    return descriptor_list       



def computeMatchingIndices(descriptor_list1, descriptor_list2, threshold=50):

    M = 3
    L = 36
  
    c_score_list = np.ndarray((len(descriptor_list1), len(descriptor_list2)))
    
    for i in range(len(descriptor_list1)):

        for j in range(len(descriptor_list2)):

            descriptor1 = descriptor_list1[i]
            descriptor2 = descriptor_list2[j]
            c_score = np.zeros((L))
            c_max = 0


            for m in range(M):
                for l in range(L):
                    for l_hat in range(L):
                        n_score = (1 - abs(descriptor1[0][m][l] - descriptor2[0][m][l_hat])) 
                        s_score = (1 - abs(descriptor1[1][m][l] - descriptor2[1][m][l_hat])) 
                        c_score[l_hat] = n_score * s_score
                    index_max = np.argmax(c_score)

                    c_max += c_score[index_max]
            
            c_score_list[i, j] = c_max
           
    
    values_to_sort = []
    indices_to_sort = []

    for i in range(len(descriptor_list1)):
        
        index = np.argmax(c_score_list[i][:])
        values_to_sort.append(c_score_list[i][index])
        indices_to_sort.append([i, index])

        
    sorted_indices = np.argsort(values_to_sort)  
    indices_to_sort = np.array(indices_to_sort)
    
    matching_indices = indices_to_sort[sorted_indices[-threshold:]]
    

    values_to_sort = np.array(values_to_sort)
    print(matching_indices)
    print(values_to_sort[sorted_indices[-threshold:]])

    

    return matching_indices





def main():

    input_file = "./data/Armadillo.ply"
    pcd = o3d.io.read_point_cloud(input_file)

    mesh = o3d.io.read_triangle_mesh(input_file)
    mesh.compute_vertex_normals()
    
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(5)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    
    flag = True

    if flag:
        
        keypoints_indices, saliency, salient_radius, non_max_radius = computeISS(pcd)
        np.save('../ISSkeypoints_indices', keypoints_indices)
        np.save('../ISSsaliency', saliency)
        np.save('../salient_radius', salient_radius)
    else:
        path_keypoints = 'ISSkeypoints_indices.npy'
        path_saliency = 'ISSsaliency.npy'
        path_salient_radius = 'salient_radius.npy'
        keypoints_indices = np.load(path_keypoints)
        saliency = np.load(path_saliency)
        salient_radius = np.load(path_salient_radius)
    print(max(saliency))
    
    
    pcd_keypoints = o3d.geometry.PointCloud()
    pcd_keypoints.points = o3d.utility.Vector3dVector(points[keypoints_indices])
    
    pcd_keypoints.paint_uniform_color([1.0, 0.0, 0.0])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    

    o3d.visualization.draw_geometries([pcd_keypoints, mesh])

if __name__ == '__main__':
    main()