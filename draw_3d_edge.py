
import open3d as o3d
import numpy as np
import time
def compute_curvature(pcd, k=30):
    curvature = []
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        _, idx, _ = kd_tree.search_knn_vector_3d(pcd.points[i], k)
        k_neighbors = np.asarray(pcd.points)[idx, :]
        covariance_matrix = np.cov(k_neighbors.T)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        curvature.append(eigenvalues[0] / np.sum(eigenvalues))
    boundary_indices = np.where(curvature > np.percentile(np.array(curvature), 90))[0]
    boundary_points = np.asarray(pcd.points)[boundary_indices]
    boundary_point_cloud = o3d.geometry.PointCloud()
    boundary_point_cloud.points = o3d.utility.Vector3dVector(boundary_points)
    return boundary_point_cloud


# 示例用法
s_time = time.time()
pcd = o3d.io.read_point_cloud(r"3d_edge_data/0416_02_hub.ply")
# o3d.visualization.draw_geometries([pcd.paint_uniform_color([1, 0, 0])])
pcd = pcd.voxel_down_sample(0.1)
pcd = compute_curvature(pcd, 20)
end_time = time.time()
print("cols:",end_time-s_time)
o3d.visualization.draw_geometries([pcd.paint_uniform_color([1, 0, 0])])

