#pipeline using SP


import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from SPkeypoints import computeSP, computeFeatureDescriptor,computeMatchingIndices
import math
from ransac import computeRANSAC



def main():

    #create = True -> create a new configuration, it takes several minutes! 
    #create = False -> load an existing configuration from the directory ./data/output/app3 
    create = False

    if create:

        pcd1, pcd2 = createScene()

        pcd1_points = np.asarray(pcd1.points)
        pcd1_normals = np.asarray(pcd1.normals)
    
        pcd2_points = np.asarray(pcd2.points)
        pcd2_normals = np.asarray(pcd2.normals)

        print('Scene has been created')
        print('Computing keypoints')

        keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, sigma_pcd1, sigma_pcd2 = computeKeypoints(pcd1, pcd2)

        print('Creating features descriptor')

        descriptor_list_pcd1 = computeFeatureDescriptor(pcd1_points, pcd1_normals, saliency_pcd1, keypoints_indices_pcd1, sigma_pcd1[1])
        descriptor_list_pcd2 = computeFeatureDescriptor(pcd2_points, pcd2_normals, saliency_pcd2, keypoints_indices_pcd2, sigma_pcd2[1])
        np.save('./data/output/app1/descriptor_list_pcd1', descriptor_list_pcd1)
        np.save('./data/output/app1/descriptor_list_pcd2', descriptor_list_pcd2)

        print('Computing matching score')
        threshold = 50 # edit this parameter to increase the numbers of matchings
        matching_indices = computeMatchingIndices(descriptor_list_pcd1, descriptor_list_pcd2, threshold)
        np.save('./data/output/app1/matching_indices', matching_indices)

        print('estimating roto-translation with RANSAC')
        # data structure:
        #   matching_points -> [index kp pcd1, index kp pcd2]
        #   kp pcd1 -> list of coords of kp pcd1
        #   kp pcd2 -> list of coords of kp pcd2

        

        R, T, error = computeRANSAC(matching_indices, pcd1_points[keypoints_indices_pcd1], pcd2_points[keypoints_indices_pcd2], k=2000)

        np.save('./data/output/app1/R', R)
        np.save('./data/output/app1/T', T)
        np.save('./data/output/app1/error_model', error)

        print(f'Ransac has found a model with an error of {error}')
        print(f'R = {R}  \n T = {T}')

        # create a new pcd 
        p_estimate = []
        for p in pcd1_points:
            p_estimate.append((np.matmul(R, np.atleast_2d(p).transpose()) + T).transpose()[0])  

        pcd_roto = o3d.geometry.PointCloud()
        pcd_roto.points = o3d.utility.Vector3dVector(p_estimate)

        pcd_roto.estimate_normals()
        pcd_roto.orient_normals_consistent_tangent_plane(k=5)

        o3d.io.write_point_cloud('./data/output/app1/pcd_roto.ply', pcd_roto)


    pcd1, pcd2, keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, sigma_pcd1, sigma_pcd2, descriptor_list_pcd1, descriptor_list_pcd2, matching_indices,R, T, error, pcd_roto = loadScene()
    
    print(f'pcd1 number of keypoints found using SP: {len(keypoints_indices_pcd1)}')
    print(f'pcd2 number of keypoints found using SP: {len(keypoints_indices_pcd2)}')
    print('descriptors computed')
    print(f'Found {len(matching_indices)} matching indices')
    print(f'Ransac has found a model with an error of {error}')
    print(f'R = {R}  \n T = {T}')
    print(f'residuals R = {R - np.array([[math.cos(90), 0, math.sin(90)], [0, 1, 0], [-math.sin(90), 0,  math.cos(90)]])} T = {T - np.array([[100], [50], [-50]])}')

    pcd1_points = np.asarray(pcd1.points)
    pcd2_points = np.asarray(pcd2.points)

    


    #plot
    
    pcd1.paint_uniform_color([1, 0.0, 0.0])
    pcd2.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_roto.paint_uniform_color([1, 0.5, 0.5])

    pcd1_keypoints = o3d.geometry.PointCloud()
    pcd1_keypoints.points = o3d.utility.Vector3dVector(pcd1_points[keypoints_indices_pcd1])
    pcd1_keypoints.paint_uniform_color([1.0, 0.75, 0.0])

    pcd2_keypoints = o3d.geometry.PointCloud()
    pcd2_keypoints.points = o3d.utility.Vector3dVector(pcd2_points[keypoints_indices_pcd2])
    pcd2_keypoints.paint_uniform_color([1.0, 0.75, 0.0])

    lines = o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1_keypoints, pcd2_keypoints, matching_indices)
    rf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)

    o3d.visualization.draw_geometries([rf, pcd1_keypoints, pcd2_keypoints, pcd1, pcd2, lines])
    o3d.visualization.draw_geometries([rf, pcd2, pcd_roto])



def loadScene():

    pcd1 = o3d.io.read_point_cloud('./data/output/app1/pcd1.ply') 
    pcd2 = o3d.io.read_point_cloud('./data/output/app1/pcd2.ply') 

    keypoints_indices_pcd1 = np.load('./data/output/app1/keypoints_indices_pcd1.npy')
    saliency_pcd1 = np.load('./data/output/app1/saliency_pcd1.npy')
    sigma_pcd1 = np.load('./data/output/app1/sigma_pcd1.npy')

    keypoints_indices_pcd2 = np.load('./data/output/app1/keypoints_indices_pcd2.npy')
    saliency_pcd2 = np.load('./data/output/app1/saliency_pcd2.npy')
    sigma_pcd2 = np.load('./data/output/app1/sigma_pcd2.npy')

    descriptor_list_pcd1 = np.load('./data/output/app1/descriptor_list_pcd1.npy')
    descriptor_list_pcd2 = np.load('./data/output/app1/descriptor_list_pcd2.npy')

    matching_indices = np.load('./data/output/app1/matching_indices.npy')

    R = np.load('./data/output/app1/R.npy')
    T = np.load('./data/output/app1/T.npy')
    error = np.load('./data/output/app1/error_model.npy')

    pcd_roto = o3d.io.read_point_cloud('./data/output/app1/pcd_roto.ply') 
    
    return [pcd1, pcd2, keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, sigma_pcd1, sigma_pcd2, descriptor_list_pcd1, descriptor_list_pcd2, matching_indices, R, T, error, pcd_roto]
   
    
def createScene():

    #input paths
    input_file1 = './data/Armadillo.ply'
    input_file2 = './data/Armadillo.ply'
    
    #load pointclouds
    pcd1 = o3d.io.read_point_cloud(input_file1)
    pcd1 = pcd1.uniform_down_sample(1)

    pcd2 = o3d.io.read_point_cloud(input_file2)
    pcd2 = pcd2.uniform_down_sample(1)

    noisy_points = []
    for p in np.asarray(pcd2.points):
        noisy_points.append(p + np.random.normal(0.0, 0.1, 3))

    pcd2.points = o3d.utility.Vector3dVector(noisy_points)


    #variables
    # alpha_pcd1 = np.radians(0)
    # beta_pcd1 = np.radians(90)
    # translation_pcd1 = np.array([100, 50, -50])
    # rotationXaxis_pcd1 = np.array([[1, 0, 0], [0, math.cos(alpha_pcd1), -math.sin(alpha_pcd1)], [0, math.sin(alpha_pcd1), math.cos(alpha_pcd1)]])
    # rotationYaxis_pcd1 = np.array([[math.cos(beta_pcd1), 0, math.sin(beta_pcd1)], [0, 1, 0], [-math.sin(beta_pcd1), 0,  math.cos(beta_pcd1)]])

    gamma_pcd2 = np.radians(0)
    beta_pcd2 = np.radians(90)
    translation_pcd2 = np.array([[100], [50], [-50]])
    rotationZaxis_pcd2 = np.array([[math.cos(gamma_pcd2), -math.sin(gamma_pcd2), 0], [math.sin(gamma_pcd2), math.cos(gamma_pcd2), 0], [0, 0, 1]])
    rotationYaxis_pcd2 = np.array([[math.cos(beta_pcd2), 0, math.sin(beta_pcd2)], [0, 1, 0], [-math.sin(beta_pcd2), 0,  math.cos(beta_pcd2)]])


    #pcd transformations
    pcd1.estimate_normals()
    pcd1.orient_normals_consistent_tangent_plane(k=5)

    pcd2_points = np.asarray(pcd2.points)
    p_estimate2 = []

    for p in pcd2_points:
        p_estimate2.append((np.matmul(rotationYaxis_pcd2, np.atleast_2d(p).transpose()) + translation_pcd2).transpose()[0])  

    pcd2.points = o3d.utility.Vector3dVector(p_estimate2)
    
    pcd2.estimate_normals()
    pcd2.orient_normals_consistent_tangent_plane(k=5)
    
    
    #save transformed mesh
    o3d.io.write_point_cloud('./data/output/app1/pcd1.ply', pcd1)
    o3d.io.write_point_cloud('./data/output/app1/pcd2.ply', pcd2)

    return [pcd1, pcd2]

def computeKeypoints(pcd1, pcd2):
    
    #compute ISS keypoints
    
    keypoints_indices_pcd1, saliency_pcd1, sigma_pcd1 = computeSP(pcd1)
    keypoints_indices_pcd2, saliency_pcd2, sigma_pcd2 = computeSP(pcd2)


    #save keypoints
    np.save('./data/output/app1/keypoints_indices_pcd1', keypoints_indices_pcd1)
    np.save('./data/output/app1/saliency_pcd1', saliency_pcd1)
    np.save('./data/output/app1/sigma_pcd1', sigma_pcd1)
    

    np.save('./data/output/app1/keypoints_indices_pcd2', keypoints_indices_pcd2)
    np.save('./data/output/app1/saliency_pcd2', saliency_pcd2)
    np.save('./data/output/app1/sigma_pcd2', sigma_pcd2)
    

    return [keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, sigma_pcd1, sigma_pcd2]


    
    
if __name__ == '__main__':
    main()


