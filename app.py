from ISSkeypoints import compute_ISS
import open3d as o3d
import numpy as np
import math

def main():
    
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

    # Read .ply file
    input_file = "Armadillo.ply"


    mesh1 = o3d.io.read_triangle_mesh(input_file)
    mesh1.translate(translation_mesh1)
    mesh1.rotate(rotationYaxis_mesh1)
    mesh1.compute_vertex_normals()

    mesh2 = o3d.io.read_triangle_mesh(input_file)
    mesh2.translate(translation_mesh2)
    mesh2.rotate(rotationYaxis_mesh2)
    mesh2.compute_vertex_normals()


    keypoints_mesh1 = compute_ISS(mesh1)
    keypoints_mesh2 = compute_ISS(mesh2)
    
    rf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
    mesh1.paint_uniform_color([0.5, 0.5, 0.5])
    mesh2.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints_mesh1.paint_uniform_color([1.0, 0.75, 0.0])
    keypoints_mesh2.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.visualization.draw_geometries([rf, mesh1, keypoints_mesh1, mesh2, keypoints_mesh2])


if __name__ == '__main__':
    main()