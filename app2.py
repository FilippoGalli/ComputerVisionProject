import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from ISSkeypoints import computeISS
import math



def main():

    create = False

    if create:
        createScene()
        print('Scene has been created')

    mesh1, pcd1, keypoints_mesh1, keypoints_pcd1, saliency_mesh1, saliency_pcd1 = loadScene()



    #feature matching
    
    threshold = 0.000001 #to be defined

    #open3d array used to plot the lines
    correspondences_indices = []
   

    #set of coordinates of the matches
    correspondences_coords1 = []
    correspondences_coords2 = []

    #normalize saliency

    saliency_tree = KDTree(np.reshape(saliency_mesh1, (len(saliency_mesh1), 1)))
  
    for i in range(len(saliency_pcd1)):
    
        
        distance, index= saliency_tree.query([[saliency_pcd1[i]]], k=1)
        
        
        if distance[0][0] < threshold:
            
            correspondences_coords1.append(keypoints_pcd1[i])
            correspondences_coords2.append(keypoints_mesh1[index[0][0]])
            
            
    for i in range(len(correspondences_coords1)):
        correspondences_indices.append((i, i))
    
    

    #plot
    mesh1.compute_vertex_normals()


    mesh1.paint_uniform_color([0.5, 0.5, 0.5])
    pcd1.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_keypoints_mesh1 = o3d.geometry.PointCloud()
    pcd_keypoints_mesh1.points = o3d.utility.Vector3dVector(keypoints_mesh1)
    pcd_keypoints_mesh1.paint_uniform_color([1.0, 0.75, 0.0])

    pcd_keypoints_pcd1 = o3d.geometry.PointCloud()
    pcd_keypoints_pcd1.points = o3d.utility.Vector3dVector(keypoints_pcd1)
    pcd_keypoints_pcd1.paint_uniform_color([0.0, 0.0, 1.0])








    matches_mesh1 = o3d.geometry.PointCloud()
    matches_mesh1.points = o3d.utility.Vector3dVector(correspondences_coords2)

    matches_pcd1 = o3d.geometry.PointCloud()
    matches_pcd1.points = o3d.utility.Vector3dVector(correspondences_coords1)

    
    lines = o3d.geometry.LineSet().create_from_point_cloud_correspondences(matches_pcd1, matches_mesh1, correspondences_indices)
    rf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)

    o3d.visualization.draw_geometries([rf, mesh1, pcd1, matches_mesh1, matches_pcd1, lines])


def loadScene():

    mesh1 = o3d.io.read_triangle_mesh('./data/output/app2/mesh1.ply') 
    pcd1 = o3d.io.read_point_cloud('./data/output/app2/pcd1.ply') 

    keypoints_mesh1 = np.load('./data/output/app2/keypoints_mesh1.npy')
    saliency_mesh1 = np.load('./data/output/app2/saliency_mesh1.npy')

    keypoints_pcd1 = np.load('./data/output/app2/keypoints_pcd1.npy')
    saliency_pcd1 = np.load('./data/output/app2/saliency_pcd1.npy')

    return [mesh1, pcd1, keypoints_mesh1, keypoints_pcd1, saliency_mesh1, saliency_pcd1]
   
    
def createScene():


    #input paths
    input_file1 = "./data/Armadillo_scans/ArmadilloSide_0.ply"
    input_file2 = "./data/Armadillo.ply"

    #load pointclouds
    pcd1 = o3d.io.read_point_cloud(input_file1)
    mesh1 = o3d.io.read_triangle_mesh(input_file2)


    #variables
    alpha_mesh1 = np.radians(0)
    beta_mesh1 = np.radians(-180)
    translation_mesh1 = np.array([100, 50, -50])
    rotationXaxis_mesh1 = np.array([[1, 0, 0], [0, math.cos(alpha_mesh1), -math.sin(alpha_mesh1)], [0, math.sin(alpha_mesh1), math.cos(alpha_mesh1)]])
    rotationYaxis_mesh1 = np.array([[math.cos(beta_mesh1), 0, math.sin(beta_mesh1)], [0, 1, 0], [-math.sin(beta_mesh1), 0,  math.cos(beta_mesh1)]])

    translation_pcd1 = np.array([-100, 50, -50])

    #mesh transformations
    mesh1.translate(translation_mesh1)
    mesh1.rotate(rotationYaxis_mesh1)

    pcd1.scale(450, np.array([0, 0, 0]))
    pcd1.translate(translation_pcd1)

    #save transformed mesh
    o3d.io.write_triangle_mesh('./data/output/app2/mesh1.ply', mesh1)
    o3d.io.write_point_cloud('./data/output/app2/pcd1.ply', pcd1)

    #compute ISS keypoints
    
    pcd_mesh1 = o3d.geometry.PointCloud()
    pcd_mesh1.points = mesh1.vertices

    keypoints_mesh1, saliency_mesh1 = computeISS(pcd_mesh1)
    keypoints_pcd1, saliency_pcd1 = computeISS(pcd1)


    #save keypoints
    np.save('./data/output/app2/keypoints_mesh1', keypoints_mesh1)
    np.save('./data/output/app2/saliency_mesh1', saliency_mesh1)

    np.save('./data/output/app2/keypoints_pcd1', keypoints_pcd1)
    np.save('./data/output/app2/saliency_pcd1', saliency_pcd1)

    
    
if __name__ == '__main__':
    main()