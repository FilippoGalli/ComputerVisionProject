import open3d as o3d
import numpy as np
import math
from ISSkeypoints import computeISS
from sklearn.neighbors import KDTree
from rototranslation import computeRototranslationParams

def main():

    create = False

    if create:
        createScene()

    mesh1, mesh2, keypoints_mesh1, keypoints_mesh2, saliency_mesh1, saliency_mesh2 = loadScene()

    #feature matching
    
    threshold = 100 #to be defined

    #open3d array used to plot the lines
    correspondences_indices = []

    #set of coordinates of the matches
    correspondences_coords1 = []
    correspondences_coords2 = []
   
    saliency_tree = KDTree(np.reshape(saliency_mesh2, (len(saliency_mesh2), 1)))
    
    for i in range(len(saliency_mesh1)):
    
        
        distance, index= saliency_tree.query([[saliency_mesh1[i]]], k=1)
        
        if distance < threshold:
            correspondences_indices.append((i, index[0][0]))
            correspondences_coords1.append(keypoints_mesh1[i])
            correspondences_coords2.append(keypoints_mesh2[index[0][0]])
        
    
    #rototranslation

    R, T = computeRototranslationParams(correspondences_coords1, correspondences_coords2)

    newPoints = []

    for p in np.asarray(mesh1.vertices):
        newPoints.append((np.matmul(R, np.atleast_2d(p).transpose()) + T).transpose()[0])
        
    
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(newPoints)

     #plot
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()

    mesh1.paint_uniform_color([0.5, 0.5, 0.5])
    mesh2.paint_uniform_color([0.5, 0.5, 0.5])

    pcd_keypoints_mesh1 = o3d.geometry.PointCloud()
    pcd_keypoints_mesh1.points = o3d.utility.Vector3dVector(keypoints_mesh1)
    pcd_keypoints_mesh1.paint_uniform_color([1.0, 0.75, 0.0])

    pcd_keypoints_mesh2 = o3d.geometry.PointCloud()
    pcd_keypoints_mesh2.points = o3d.utility.Vector3dVector(keypoints_mesh2)
    pcd_keypoints_mesh2.paint_uniform_color([0.0, 0.0, 1.0])

    lines = o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd_keypoints_mesh1, pcd_keypoints_mesh2, correspondences_indices)
    rf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)

    o3d.visualization.draw_geometries([pcd3, mesh1, mesh2])
    



    
def loadScene():

    mesh1 = o3d.io.read_triangle_mesh('./data/mesh1.ply') 
    mesh2 = o3d.io.read_triangle_mesh('./data/mesh2.ply') 

    keypoints_mesh1 = np.load('./data/keypoints_mesh1.npy')
    keypoints_mesh2 = np.load('./data/keypoints_mesh2.npy')

    saliency_mesh1 = np.load('./data/saliency_mesh1.npy')
    saliency_mesh2 = np.load('./data/saliency_mesh2.npy')

    return [mesh1, mesh2, keypoints_mesh1, keypoints_mesh2, saliency_mesh1, saliency_mesh2]

   


def createScene():


    #input paths
    input_file = "./data/Armadillo.ply"

    #load mesh
    mesh1 = o3d.io.read_triangle_mesh(input_file)
    mesh2 = o3d.io.read_triangle_mesh(input_file)

    #variables
    alpha_mesh1 = np.radians(0)
    beta_mesh1 = np.radians(-180)
    translation_mesh1 = np.array([100, 50, -50])
    rotationXaxis_mesh1 = np.array([[1, 0, 0], [0, math.cos(alpha_mesh1), -math.sin(alpha_mesh1)], [0, math.sin(alpha_mesh1), math.cos(alpha_mesh1)]])
    rotationYaxis_mesh1 = np.array([[math.cos(beta_mesh1), 0, math.sin(beta_mesh1)], [0, 1, 0], [-math.sin(beta_mesh1), 0,  math.cos(beta_mesh1)]])

    alpha_mesh2 = np.radians(0)
    beta_mesh2 = np.radians(-90)
    translation_mesh2 = np.array([-100, 50, -50])
    rotationXaxis_mesh2 = np.array([[1, 0, 0], [0, math.cos(alpha_mesh2), -math.sin(alpha_mesh2)], [0, math.sin(alpha_mesh2), math.cos(alpha_mesh2)]])
    rotationYaxis_mesh2 = np.array([[math.cos(beta_mesh2), 0, math.sin(beta_mesh2)], [0, 1, 0], [-math.sin(beta_mesh2), 0,  math.cos(beta_mesh2)]])

    #mesh transformations
    mesh1.translate(translation_mesh1)
    mesh1.rotate(rotationYaxis_mesh1)
    
    mesh2.translate(translation_mesh2)
    mesh2.rotate(rotationYaxis_mesh2)
    
    #save transformed mesh
    o3d.io.write_triangle_mesh('./data/mesh1.ply', mesh1)
    o3d.io.write_triangle_mesh('./data/mesh2.ply', mesh2)


    #compute ISS keypoints
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = mesh1.vertices
    keypoints_pcd1, saliency_pcd1 = computeISS(pcd1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = mesh2.vertices
    keypoints_pcd2, saliency_pcd2 = computeISS(pcd2)

    #save keypoints
    np.save('./data/keypoints_mesh1', keypoints_pcd1)
    np.save('./data/saliency_mesh1', saliency_pcd1)

    np.save('./data/keypoints_mesh2', keypoints_pcd2)
    np.save('./data/saliency_mesh2', saliency_pcd2)




if __name__ == '__main__':
    main()



