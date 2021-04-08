import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from ISSkeypoints import computeISS, computeFeatureDescriptor,computeMatchingIndices
import math



def main():

    create = True

    if create:
        pcd1, pcd2 = createScene()

        pcd1_points = np.asarray(pcd1.points)
        pcd1_normals = np.asarray(pcd1.normals)
    
        pcd2_points = np.asarray(pcd2.points)
        pcd2_normals = np.asarray(pcd2.normals)

        print('Scene has been created')

        keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, salient_radius_pcd1, salient_radius_pcd2, non_max_radius_pcd1, non_max_radius_pcd2 = computeKeypoints(pcd1, pcd2)

        print('Creating features descriptor')

        descriptor_list_pcd1 = computeFeatureDescriptor(pcd1_points, pcd1_normals, saliency_pcd1, keypoints_indices_pcd1, salient_radius_pcd1)
        descriptor_list_pcd2 = computeFeatureDescriptor(pcd2_points, pcd2_normals, saliency_pcd2, keypoints_indices_pcd2, salient_radius_pcd2)
        np.save('./data/output/app3/descriptor_list_pcd1', descriptor_list_pcd1)
        np.save('./data/output/app3/descriptor_list_pcd2', descriptor_list_pcd2)

        print('Computing matching score')
        threshold = 30
        matching_indices = computeMatchingIndices(descriptor_list_pcd1, descriptor_list_pcd2, threshold)
        np.save('./data/output/app3/matching_indices', matching_indices)

    print('Load data')
    pcd1, pcd2, keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, salient_radius_pcd1, salient_radius_pcd2, descriptor_list_pcd1, descriptor_list_pcd2, matching_indices   = loadScene()

    pcd1_points = np.asarray(pcd1.points)

    pcd2_points = np.asarray(pcd2.points)

    

    #plot
    
    pcd1.paint_uniform_color([0.5, 0.5, 0.5])
    pcd2.paint_uniform_color([0.5, 0.5, 0.5])

    pcd1_keypoints = o3d.geometry.PointCloud()
    pcd1_keypoints.points = o3d.utility.Vector3dVector(pcd1_points[keypoints_indices_pcd1])
    pcd1_keypoints.paint_uniform_color([1.0, 0.75, 0.0])

    pcd2_keypoints = o3d.geometry.PointCloud()
    pcd2_keypoints.points = o3d.utility.Vector3dVector(pcd2_points[keypoints_indices_pcd2])
    pcd2_keypoints.paint_uniform_color([1.0, 0.75, 0.0])

    lines = o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1_keypoints, pcd2_keypoints, matching_indices)
    rf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)

    o3d.visualization.draw_geometries([rf, pcd1_keypoints, pcd2_keypoints, pcd1, pcd2, lines])



def loadScene():

    pcd1 = o3d.io.read_point_cloud('./data/output/app3/pcd1.ply') 
    pcd2 = o3d.io.read_point_cloud('./data/output/app3/pcd2.ply') 

    keypoints_indices_pcd1 = np.load('./data/output/app3/keypoints_indices_pcd1.npy')
    saliency_pcd1 = np.load('./data/output/app3/saliency_pcd1.npy')
    salient_radius_pcd1 = np.load('./data/output/app3/salient_radius_pcd1.npy')

    keypoints_indices_pcd2 = np.load('./data/output/app3/keypoints_indices_pcd2.npy')
    saliency_pcd2 = np.load('./data/output/app3/saliency_pcd2.npy')
    salient_radius_pcd2 = np.load('./data/output/app3/salient_radius_pcd2.npy')

    descriptor_list_pcd1 = np.load('./data/output/app3/descriptor_list_pcd1.npy')
    descriptor_list_pcd2 = np.load('./data/output/app3/descriptor_list_pcd2.npy')

    matching_indices = np.load('./data/output/app3/matching_indices.npy')

    return [pcd1, pcd2, keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, salient_radius_pcd1, salient_radius_pcd2, descriptor_list_pcd1, descriptor_list_pcd2, matching_indices]
   
    
def createScene():

    #input paths
    input_file1 = "./data/Armadillo.ply"
    input_file2 = "./data/Armadillo_scans/ArmadilloStandFlip_0.ply"
    
    #load pointclouds
    pcd1 = o3d.io.read_point_cloud(input_file1)
    pcd1 = pcd1.uniform_down_sample(2)
    pcd1.estimate_normals()
    pcd1.orient_normals_consistent_tangent_plane(k=5)
  
    
    pcd2 = o3d.io.read_point_cloud(input_file2)

    points = np.asarray(pcd2.points)

    #pcd2 = pcd2.uniform_down_sample(5)
    pcd2.estimate_normals()
    pcd2.orient_normals_consistent_tangent_plane(k=5)

    #variables
    alpha_pcd1 = np.radians(0)
    beta_pcd1 = np.radians(90)
    translation_pcd1 = np.array([100, 50, -50])
    rotationXaxis_pcd1 = np.array([[1, 0, 0], [0, math.cos(alpha_pcd1), -math.sin(alpha_pcd1)], [0, math.sin(alpha_pcd1), math.cos(alpha_pcd1)]])
    rotationYaxis_pcd1 = np.array([[math.cos(beta_pcd1), 0, math.sin(beta_pcd1)], [0, 1, 0], [-math.sin(beta_pcd1), 0,  math.cos(beta_pcd1)]])

    gamma_pcd2 = np.radians(90)
    beta_pcd2 = np.radians(90)
    translation_pcd2 = np.array([-200, 59, -50])
    rotationZaxis_pcd2 = np.array([[math.cos(gamma_pcd2), -math.sin(gamma_pcd2), 0], [math.sin(gamma_pcd2), math.cos(gamma_pcd2), 0], [0, 0, 1]])
    rotationYaxis_pcd2 = np.array([[math.cos(beta_pcd2), 0, math.sin(beta_pcd2)], [0, 1, 0], [-math.sin(beta_pcd2), 0,  math.cos(beta_pcd2)]])


    #pcd transformations
    pcd1.rotate(rotationYaxis_pcd1)
    pcd1.translate(translation_pcd1)
   

    pcd2.scale(700, np.array([0, 0, 0]))
    pcd2.rotate(rotationZaxis_pcd2)
    pcd2.rotate(rotationYaxis_pcd2)
    pcd2.translate(translation_pcd2)
    

    #save transformed mesh
    o3d.io.write_point_cloud('./data/output/app3/pcd1.ply', pcd1)
    o3d.io.write_point_cloud('./data/output/app3/pcd2.ply', pcd2)

    return [pcd1, pcd2]

def computeKeypoints(pcd1, pcd2):
    
    #compute ISS keypoints
    
    keypoints_indices_pcd1, saliency_pcd1, salient_radius_pcd1, non_max_radius_pcd1 = computeISS(pcd1)
    keypoints_indices_pcd2, saliency_pcd2, salient_radius_pcd2, non_max_radius_pcd2 = computeISS(pcd2)


    #save keypoints
    np.save('./data/output/app3/keypoints_indices_pcd1', keypoints_indices_pcd1)
    np.save('./data/output/app3/saliency_pcd1', saliency_pcd1)
    np.save('./data/output/app3/salient_radius_pcd1', salient_radius_pcd1)
    np.save('./data/output/app3/non_max_radius_pcd1', non_max_radius_pcd1)


    np.save('./data/output/app3/keypoints_indices_pcd2', keypoints_indices_pcd2)
    np.save('./data/output/app3/saliency_pcd2', saliency_pcd2)
    np.save('./data/output/app3/salient_radius_pcd2', salient_radius_pcd2)
    np.save('./data/output/app3/non_max_radius_pcd2', non_max_radius_pcd2)

    return [keypoints_indices_pcd1, keypoints_indices_pcd2, saliency_pcd1, saliency_pcd2, salient_radius_pcd1, salient_radius_pcd2, non_max_radius_pcd1, non_max_radius_pcd2]


    
    
if __name__ == '__main__':
    main()


