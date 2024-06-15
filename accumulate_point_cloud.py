import numpy as np
from PIL import Image
import json
import torch
from unidepth.models import UniDepthV1, UniDepthV2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import open3d as o3d

#model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Function to remove extra points or colors to match the number of points and colors
def remove_extra(data, target_length):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        ratio = len(data) / target_length
        indices = [int(i * ratio) for i in range(target_length)]
        return [data[i] for i in indices]
    return data
with open(("/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/data_test/transforms_ego_cam_coord.json"), "r") as f:
        input_data = json.load(f)
K = np.zeros((3,3)) # (C,3,3)
           # K[0,0] = input_data['fl_x']             #Focus_length_x = ImageSizeX /(2 * tan(CameraFOV * π / 360))
K[0,0] = input_data['img_size'][0] / (2*np.tan(input_data['fov'] * np.pi / 360))
           # K[1,1] = input_data['fl_y']             #Focus_length_y = ImageSizey /(2 * tan(CameraFOV * π / 360))
K[1,1] = input_data['img_size'][0] / (2*np.tan(input_data['fov'] * np.pi / 360))  #fl_x = fl_y = f
        #K = [[f, 0, Cu],
        #     [0, f, Cv],
        #     [0, 0, 1 ]]
K[2,2] = 1
            # K[0,2] = input_data['cx']               #ImageSizeX / 2
K[0,2] = input_data['img_size'][0] / 2
            # K[1,2] = input_data['cy']               #ImageSizeY / 2
K[1,2] = input_data['img_size'][1] / 2
# Convert K from numpy array to PyTorch tensor
K = torch.tensor(K).float() #test
intrinsic_matrix_4x4 = np.eye(4)
intrinsic_matrix_4x4[:3,:3] = K
#print("intrinsic_matrix_4x4----------------------")
#print(intrinsic_matrix_4x4)
# COnvert K to intrinsix matrix 
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(input_data['img_size'][0], input_data['img_size'][1], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
#print("intrinsic matrix----------", intrinsic)
extrinsics = {}
folder_img_path = '/home/ubuntu/Workspace/phat-intern-dev/VinAI/6Img-to-3D-at-VinAI/data_test'
for filename in os.listdir(folder_img_path):
    if filename.endswith(".png"):
        fname = filename.split(".")[0]
        print(fname)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix = np.array(input_data['transform'][f'{fname}'])
        extrinsics[fname] = extrinsic_matrix

        print(extrinsic_matrix)
        print(extrinsic_matrix.shape)
        print("-------------------------------")
print(extrinsics)


def estimate_depth(image_path, model):
    print(image_path)
    image = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1).float() / 255.0 # C, H, W
    with torch.no_grad():
        predictions = model.infer(image.to(device), K.to(device))
    depth = predictions["depth"]
    xyz = predictions["points"]
    xyz = xyz.cpu().squeeze().permute(1, 2, 0).reshape(-1, 3).numpy()
    depth_map = depth.cpu().squeeze().numpy()
    return depth_map, image, xyz
# Paths to your 6 images

image_paths = [os.path.join(folder_img_path, fname) for fname in os.listdir(folder_img_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]




points_list = []
points_list_depth = []
colors_list = []

for idx, fname in enumerate(image_paths):
    
    name = fname.split("/")[-1].split(".")[0]
    
    print("------------")
    print(name)
    depth_map, image, xyz = estimate_depth(image_paths[idx], model)
    print("depth_map")
    print(depth_map)
    print(depth_map.shape)
    # Ussing depth map----------------------------
    #convert to world coordinate
    # ones = np.ones((depth_map.shape[0], 1))
    # depth_map_homogeneous = np.hstack((depth_map, ones))
    # print("depth_map_homogeneous", depth_map_homogeneous.shape)
    # depth_map_homogeneous_world = np.linalg.inv(extrinsics[f'{name}']) @ np.linalg.inv(intrinsic_matrix_4x4) @ depth_map_homogeneous.T
    # depth_map_homogeneous_world = depth_map_homogeneous_world[:3, :].T
    # depth_map = depth_map_homogeneous_world

    depth_o3d = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))  # Scaling depth to millimeters
    point_cloud_depth = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic, extrinsic = np.linalg.inv(extrinsics[f'{name}']))
    points_depth = np.asarray(point_cloud_depth.points)
    print("point_cloud")
    print(points_depth)
    print(points_depth.shape)
    #using point cloud camera coord
    # ones = np.ones((points_depth.shape[0], 1))
    # points_depth_homogeneous = np.hstack((points_depth, ones))
    # print("points_depth_homogeneous", points_depth_homogeneous.shape)
    # point_cloud = np.linalg.inv(extrinsics[f'{name}']) @ points_depth_homogeneous.T
    # #point_cloud = extrinsics[f'{name}'] @ points_depth_homogeneous.T
    # points = point_cloud[:3, :].T
    # print("point cloud xyz------------")
    # print(type(points))
    # print(points.shape)
    

    # Extract colors from the original image
    colors = image.permute(1, 2, 0).reshape(-1, 3).numpy()
    #if name == "input_camera_6":

    print("depth--------",points_depth.shape)
    # print("points--------",points.shape)
    print("color---------",colors.shape)
    if points_depth.shape[0] != colors.shape[0]:
        print(f"Number of points {points_depth.shape[0]} does not match number of colors {colors.shape[0]} for image {name}")
        min_length = min(points_depth.shape[0], colors.shape[0])
        points_depth = remove_extra(points_depth, min_length)
        colors = remove_extra(colors, min_length)

    # points_list.append(points)
    points_list_depth.append(points_depth)
    colors_list.append(colors)
# Concatenate depth maps horizontally (or use another method to combine them)
# Concatenate points and colors from both images
# points = np.concatenate(points_list, axis=0)
points_depth = np.concatenate(points_list_depth, axis=0)
colors = np.concatenate(colors_list, axis=0)


colors3d = o3d.utility.Vector3dVector(colors)

# # Create a PointCloud object with colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_depth)
pcd.colors = colors3d


# pcloud_3d = o3d.utility.Vector3dVector(points)
# pc = o3d.geometry.PointCloud()
# pc.points = pcloud_3d
# pc.colors = colors3d

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

# # Function to create a grid on the Oxy plane
# def create_grid(size=10, step=1):
#     grid_lines = []
#     for i in range(-size, size + 1, step):
#         grid_lines.append(([i, -size, 0], [i, size, 0]))
#         grid_lines.append(([-size, i, 0], [size, i, 0]))
#     grid = o3d.geometry.LineSet()
#     points = np.array([line[0] for line in grid_lines] + [line[1] for line in grid_lines])
#     lines = [[2 * i, 2 * i + 1] for i in range(len(grid_lines))]
#     grid.points = o3d.utility.Vector3dVector(points)
#     grid.lines = o3d.utility.Vector2iVector(lines)
#     return grid

# # Create grid
# grid = create_grid()

# # Convert the coordinate frame to a point cloud
# coordinate_frame_points = np.asarray(coordinate_frame.vertices)
# coordinate_frame_colors = np.asarray(coordinate_frame.vertex_colors)

# # Append coordinate frame points and colors to the point cloud
# all_points = np.concatenate((points, coordinate_frame_points), axis=0)
# all_colors = np.concatenate((colors, coordinate_frame_colors), axis=0)

# # Append grid points and lines to the point cloud
# grid_points = np.asarray(grid.points)
# grid_colors = np.full(grid_points.shape, [0.5, 0.5, 0.5])  # Set grid color to gray

# all_points = np.concatenate((all_points, grid_points), axis=0)
# all_colors = np.concatenate((all_colors, grid_colors), axis=0)

# # Create a combined PointCloud object
# combined_pcd = o3d.geometry.PointCloud()
# combined_pcd.points = o3d.utility.Vector3dVector(all_points)
# combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)


#o3d.io.write_point_cloud("point_cloud_with_colors_using_depth_unidepth_img_1_4.ply", pcd)
o3d.io.write_point_cloud("/home/ubuntu/Workspace/phat-intern-dev/VinAI/Unidepthv2/point_cloud_with_colors_using_depth_unidepth_full_not_world_coord_inv_extrinsic.ply", pcd)