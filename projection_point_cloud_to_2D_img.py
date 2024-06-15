from interpolated_poses import get_interpolated_poses
import open3d as o3d
import numpy as np
import cv2
import json

# Load PLY file
ply_file_path = '/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/point_cloud_with_colors_using_depth_unidepth_full_not_world_coord_inv_extrinsic.ply'
point_cloud = o3d.io.read_point_cloud(ply_file_path)
print(point_cloud)

intrinsic_matrix = np.array([[800, 0, 800],
                             [0, 800, 464],
                             [0, 0, 1]], np.float32)

def transform_point_cloud(point_cloud, transformation_matrix, intrinsic_matrix, output_file_rgb, output_file_depth, width=1600, height=928):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points_homogeneous = transformation_matrix @ points_homogeneous.T
    transformed_points = transformed_points_homogeneous[:3].T

    # Project the 3D points to 2D points
    dist_coeffs = np.zeros((5, 1), np.float32)
    rvec = np.zeros((3, 1), np.float32)
    tvec = np.zeros((3, 1), np.float32)
    points_2d, _ = cv2.projectPoints(transformed_points, rvec, tvec, intrinsic_matrix, dist_coeffs)
    points_2d = np.round(points_2d)

    depth_list = [[[] for _ in range(width)] for _ in range(height)]
    color_list = [[[] for _ in range(width)] for _ in range(height)]

    for i in range(len(points_2d)):
        u, v = points_2d[i][0]
        u,v = int(u), int(v)
        if 0 <= u < width and 0 <= v < height:
            depth_value = transformed_points[i][2]
            depth_list[v][u].append(depth_value)
            color_list[v][u].append(colors[i] * 255)  # Set all three channels (B, G, R)

    # Resolve final depth and color based on nearest depth
    final_rgb_img = np.zeros((height, width, 3), dtype=np.float32)
    final_depth_map = np.full((height, width), np.inf, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            if depth_list[y][x]:
                # Find the nearest depth and its corresponding color
                min_depth_idx = np.argmax(depth_list[y][x])
                final_depth_map[y][x] = depth_list[y][x][min_depth_idx]
                final_rgb_img[y][x] = color_list[y][x][min_depth_idx]

    # Convert BGR to RGB format for saving
    im_rgb = cv2.cvtColor(final_rgb_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_file_rgb + ".png", im_rgb)
    cv2.imwrite(output_file_depth + ".png", final_depth_map)

def load_extrinsics_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['transform']

# Path to the JSON file containing extrinsics
json_file_path = '/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/data_test/transforms_ego_cam_coord.json'
extrinsics_data = load_extrinsics_from_json(json_file_path)

steps = 10
for i in range(1, len(extrinsics_data) + 1):
    name_a = f"input_camera_{i}"
    if i == len(extrinsics_data):
        name_b = f"input_camera_1"
    else:
        name_b = f"input_camera_{i + 1}"
    print("-----------------------------------")
    print(f"view_{name_a}_view_{name_b}")
    pose_a = np.linalg.inv(extrinsics_data[name_a])
    pose_b = np.linalg.inv(extrinsics_data[name_b])
    interpolated_poses = get_interpolated_poses(pose_a, pose_b, steps)
    for j in range(steps):
        output_file_prefix = f"pair_view_{name_a}_view_{name_b}_pose_{j}"
        print(output_file_prefix)
        output_file_rgb = f"/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/folder_projection_2d/rgb_img/view2/{output_file_prefix}"
        output_file_depth = f"/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/folder_projection_2d/depth_img/view2/{output_file_prefix}"
        transform_point_cloud(point_cloud, interpolated_poses[j], intrinsic_matrix, output_file_rgb, output_file_depth)