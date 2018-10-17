import open3d
import numpy as np
import os.path as osp
from transforms3d.quaternions import quat2mat

def read_xyz_file(file):
    with open(file, 'r') as f:
        d = np.array([l.strip("\n").split(" ") for l in f.readlines()], dtype=np.float32)
    return d


def backproject_camera(im_depth, meta_data):

    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # get intrinsic matrix
    K = meta_data['intrinsic_matrix']
    K = np.matrix(K)
    Kinv = np.linalg.inv(K)
    # if cfg.FLIP_X:
    #     Kinv[0, 0] = -1 * Kinv[0, 0]
    #     Kinv[0, 2] = -1 * Kinv[0, 2]

    # compute the 3D points        
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = Kinv * x2d.transpose()

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

    # mask
    # index = np.where(im_depth.flatten() == 0)
    # X[:,index] = np.nan
    # pts = []
    # for i in xrange(X.shape[-1]):
    #     X_i = X[:,i]
    #     if np.all(X_i!=0):
    #         pts.append([float(X_i[0]),float(X_i[1]),float(X_i[2])])

    return np.array(X)


def render_object_pose(im, depth, meta_data, pose_data, object_model_dir):
    """
    im: rgb image of the scene
    depth: depth image of the scene
    meta_data: dict({'intrinsic_matrix': K, 'factor_depth': })
    pose_data: [{"name": "004_sugar_box", "pose": 3x4 or 4x4 matrix}, {...}, ]
    object_model_dir: directory where the object .xyz files are stored
    """
    if len(pose_data) == 0:
        return 

    rgb = im.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    X = backproject_camera(depth, meta_data)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = open3d.PointCloud(); 
    scene_cloud.points = open3d.Vector3dVector(X.T)
    scene_cloud.colors = open3d.Vector3dVector(cloud_rgb)

    if len(pose_data) == 0:
        open3d.draw_geometries([scene_cloud])
        return 

    for pd in pose_data:
        object_name = pd["name"]
        object_pose = pd["pose"]
        object_cloud_file = osp.join(object_model_dir,object_name,"points.xyz")
        object_pose_matrix4f = np.identity(4)
        object_pose = np.array(object_pose)
        if object_pose.shape == (4,4):
            object_pose_matrix4f = object_pose
        elif object_pose.shape == (3,4):
            object_pose_matrix4f[:3,:] = object_pose
        elif len(object_pose) == 7:
            object_pose_matrix4f[:3,:3] = quat2mat(object_pose[:4])
            object_pose_matrix4f[:3,-1] = object_pose[4:]
        else:
            print("[WARN]: Object pose for %s is not of shape (4,4) or (3,4) or 1-d quat (7), skipping..."%(object_name))
            continue
        # object_pose_T = object_pose[:,3]
        # object_pose_R = object_pose[:,:3]

        object_pts3d = read_xyz_file(object_cloud_file)
        object_cloud = open3d.PointCloud(); 
        object_cloud.points = open3d.Vector3dVector(object_pts3d)
        object_cloud.transform(object_pose_matrix4f)

        print("Showing %s"%(object_name))
        open3d.draw_geometries([scene_cloud, object_cloud])


if __name__ == '__main__':
    import sys        
    import cv2
    import json
    # import tensorflow.contrib.slim as slim  # SEGFAULTS when used with open3d AND cv2

    root_data_dir = "./data"
    rgb_file = osp.join(root_data_dir, "demo_images/000001-color.png")
    depth_file = osp.join(root_data_dir, "demo_images/000001-depth.png")
    pose_file = osp.join(root_data_dir, "demo_images/000001-pred_pose.json")
    object_model_dir = osp.join(root_data_dir, "LOV/models")

    rgb = cv2.imread(rgb_file)

    if rgb is None:
        print("Could not find %s"%(rgb_file))
        sys.exit(1)

    depth  = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print("Could not find %s"%(depth_file))
        sys.exit(1)

    # construct meta data
    # K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    # meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 10000.0})

    with open(pose_file, 'r') as f: 
        j_data = json.load(f)
        pose_data = j_data['poses']
        meta_data = j_data['meta']

    render_object_pose(rgb, depth, meta_data, pose_data, object_model_dir)
