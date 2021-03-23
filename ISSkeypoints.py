import open3d as o3d
import time
import numpy as np

input_file = "Armadillo.ply"
mesh = o3d.io.read_triangle_mesh(input_file)

pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices

tic = time.time()
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
print(type(keypoints))
toc = 1000 * (time.time() - tic)
print("ISS Computation took {:.0f} [ms]".format(toc))

mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])
keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([keypoints, mesh])