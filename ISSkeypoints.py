from sklearn.neighbors import KDTree
import open3d as o3d
import numpy as np
import time


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
    keypoints_index = []
    saliency_keypoint = []
    
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
                keypoints_index.append(i)
                saliency_keypoint.append(saliency[i])

    toc = 1000 * (time.time() - tic)

    print("SP Computation took {:.0f} [s]".format(toc/1000))
    print(f'number of keypoints found: {len(points[keypoints_index])}')

    return [points[keypoints_index], saliency_keypoint]
                  
        

def main():

    input_file = "./data/Armadillo_scans/ArmadilloOnHeadMultiple_0.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = mesh.vertices
    print(len(pcd.points))
    flag = True

    if flag:

        keypoints, saliency = computeISS(pcd)
        np.save('ISSkeypoints', keypoints)
        np.save('ISSsaliency', saliency)
    else:
        path_keypoints = './data/output/ISSkeypoints.npy'
        path_saliency = './data/output/ISSsaliency.npy'
        keypoints = np.load(path_keypoints)
        saliency = np.load(path_saliency)
    
    pcd_keypoints = o3d.geometry.PointCloud()
    pcd_keypoints.points = o3d.utility.Vector3dVector(keypoints)
    
    pcd_keypoints.paint_uniform_color([1.0, 0.0, 0.0])
    pcd.paint_uniform_color([0.0, 0.0, 1.0])
    #mesh.compute_vertex_normals()
    #mesh.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd_keypoints, pcd])

if __name__ == '__main__':
    main()