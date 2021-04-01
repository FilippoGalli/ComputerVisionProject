import open3d as o3d
import numpy as np


def createOctahedron(c=[0.0, 0.0, 0.0], r= 1.0):

    vertices = [[c[0] + r, c[1], c[2]], [c[0], c[1] + r, c[2]], [c[0], c[1], c[2] + r], [c[0] + (-r), c[1], c[2]], [c[0], c[1] + (-r), c[2]], [c[0], c[1], c[2] + (-r)]]
    indices = [[0, 1], [0, 2], [0, 4], [0, 5], [3, 2], [3, 4], [3, 1], [3, 5], [2, 1], [2, 4], [5, 1], [5, 4]]
    triangles=[[0, 4], [0, 3], [3, 7], [4, 7], [1, 2], [2, 6], [1, 5], [5, 6], [0, 1], [2, 3], [4, 5], [6, 7]]
    return [vertices, indices, triangles]


def sphericalGrid(c, r):

    vertices, indices, triangles = createOctahedron(c, r)

    def computeMidPoint(a, b):

        if abs(a) >= abs(b):
            return b + (a - b) / 2
        return a + (b - a) / 2

    newVertices = []
    newIndices = []
    pos = len(vertices)
    for i in range(len(indices)):
        
        a = indices[i][0]
        b = indices[i][1]
        
        x = computeMidPoint(vertices[a][0], vertices[b][0]) 
        y = computeMidPoint(vertices[a][1], vertices[b][1])
        z = computeMidPoint(vertices[a][2], vertices[b][2])
        
        vector = [x - c[0], y - c[1], z - c[2]]

        norm_vector = vector / np.linalg.norm(vector)

        x = c[0] + r * norm_vector[0]
        y = c[1] + r * norm_vector[1]
        z = c[2] + r * norm_vector[2]

        vertices.append([x, y, z])
        newIndices.append([a, pos])
        newIndices.append([b, pos])
        
        newVertices.append([pos, triangles[i]])
        pos += 1

    for i in range(len(newVertices)-1):
        for j in range(i+1, len(newVertices)):

            triangleA = newVertices[i][1]
            triangleB = newVertices[j][1]
            if  len(list(set(triangleA).intersection(triangleB))) != 0:
               newIndices.append([newVertices[i][0], newVertices[j][0]])
    
    return vertices, newIndices
        

def main():

    center = [5.0, 0.0, 0.0]
    radius = 3.0
    pcd = o3d.geometry.PointCloud()
    vertices, indices = sphericalGrid(center, radius)

    sphere = o3d.geometry.PointCloud()
    sphere.points = o3d.utility.Vector3dVector(vertices)
  
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(indices))
    pcd.points = o3d.utility.Vector3dVector(vertices)

    o3d.visualization.draw_geometries([pcd, lines])


if __name__ == '__main__':

    main()
    