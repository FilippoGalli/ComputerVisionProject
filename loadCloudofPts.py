import open3d as o3d
import numpy as np
import math
from ISSkeypoints import computeISS

#path = './data/Armadillo_scans/ArmadilloSide2_90.ply'
#path = './data/Armadillo_scans/ArmadilloStandFlip_0.ply'

#pcd2 = o3d.io.read_point_cloud(path)
pcd1 = o3d.io.read_point_cloud('./data/Armadillo.ply')

keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd1)

kp1 = np.asarray(keypoints.points)

keypoints_indices, saliency,  salient_radius, non_max_radius = computeISS(pcd1)


p = np.array(pcd1.points)
kp2 = np.asarray(p[keypoints_indices])


for i in range(len(kp1)):
    for j in range(3):
        if kp1[i][j] - kp2[i][j] != 0:
            print('fuck you')

print(kp1[0])
print(kp2[0])




#pcd1 = pcd1.uniform_down_sample(5)


pcd1.paint_uniform_color([0.5, 0.5, 0.5])
# pcd2.paint_uniform_color([0.5, 0.5, 0.5])


 #variables
alpha_pcd1 = np.radians(0)
beta_pcd1 = np.radians(90)
translation_pcd1 = np.array([100, 50, -50])
rotationXaxis_pcd1 = np.array([[1, 0, 0], [0, math.cos(alpha_pcd1), -math.sin(alpha_pcd1)], [0, math.sin(alpha_pcd1), math.cos(alpha_pcd1)]])
rotationYaxis_pcd1 = np.array([[math.cos(beta_pcd1), 0, math.sin(beta_pcd1)], [0, 1, 0], [-math.sin(beta_pcd1), 0,  math.cos(beta_pcd1)]])


gamma_pcd2 = np.radians(90)
beta_pcd2 = np.radians(30)
translation_pcd2 = np.array([-100, -20, -50])
rotationZaxis_pcd2 = np.array([[math.cos(gamma_pcd2), -math.sin(gamma_pcd2), 0], [math.sin(gamma_pcd2), math.cos(gamma_pcd2), 0], [0, 0, 1]])
rotationYaxis_pcd2 = np.array([[math.cos(beta_pcd2), 0, math.sin(beta_pcd2)], [0, 1, 0], [-math.sin(beta_pcd2), 0,  math.cos(beta_pcd2)]])

#pcd transformations
pcd1.rotate(rotationYaxis_pcd1)
pcd1.translate(translation_pcd1)


# pcd2.scale(700, np.array([0, 0, 0]))
# pcd2.rotate(rotationZaxis_pcd2)
# pcd2.rotate(rotationYaxis_pcd2)
# pcd2.translate(translation_pcd2)

pcd1_keypoints = o3d.geometry.PointCloud()
pcd1_keypoints.points = o3d.utility.Vector3dVector(p[keypoints_indices])
pcd1_keypoints.paint_uniform_color([1.0, 0.75, 0.0])
keypoints.paint_uniform_color([1.0, 0.0, 0.0])



o3d.visualization.draw_geometries([keypoints, pcd1_keypoints])