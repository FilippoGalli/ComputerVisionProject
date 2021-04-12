import open3d as o3d
import numpy as np
import math
from ISSkeypoints import computeISS

path = './data/Armadillo_scans/ArmadilloBack_300.ply'

# box_points = [[-63.50012207, 30.0,   -57.71451569],
#  [ 63.50013351, 30.0,   -57.71451569],
#  [-63.50012207,  97.08042145, -57.71451569],
#  [-63.50012207, 30.0,    57.71400452],
#  [ 63.50013351,  97.08042145,  57.71400452],
#  [-63.50012207,  97.08042145,  57.71400452],
#  [ 63.50013351, 30.0,   57.71400452],
#  [ 63.50013351,  97.08042145, -57.71451569]]

pcd1 = o3d.io.read_point_cloud(path)

# # box = pcd1.get_axis_aligned_bounding_box()

# box = o3d.geometry.AxisAlignedBoundingBox()

# box = box.create_from_points(o3d.utility.Vector3dVector(box_points))

# pcd1 = pcd1.crop(box)
# o3d.io.write_point_cloud('half_armadillo.ply', pcd1)

pcd1.paint_uniform_color([0.5, 0.5, 0.5])


o3d.visualization.draw_geometries([pcd1])