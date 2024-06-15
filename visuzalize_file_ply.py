import open3d as o3d

# Path to your .ply file
ply_file_path = '/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/point_cloud_with_colors_using_depth_unidepth_full_not_world_coord_inv_extrinsic.ply'
# Read the .ply file
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Print basic information about the point cloud
print(point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
