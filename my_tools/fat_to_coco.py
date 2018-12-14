import numpy as np
import cv2
from transforms3d.quaternions import mat2quat, quat2mat
import open3d

import pywavefront  # for loading .obj files

def get_camera_settings_intrinsic_matrix(camera_settings):
    intrinsic_settings = camera_settings['intrinsic_settings']
    intrinsic_matrix = np.identity(3)
    intrinsic_matrix[0,0] = intrinsic_settings['fx']
    intrinsic_matrix[1,1] = intrinsic_settings['fy']
    intrinsic_matrix[0,2] = intrinsic_settings['cx']
    intrinsic_matrix[1,2] = intrinsic_settings['cy']
    return intrinsic_matrix

def create_cloud(points, normals=[], colors=[], T=None):
    cloud = open3d.PointCloud()
    cloud.points = open3d.Vector3dVector(points)
    if len(normals) > 0:
        assert len(normals) == len(points)
        cloud.normals = open3d.Vector3dVector(normals)
    if len(colors) > 0:
        assert len(colors) == len(points)
        cloud.colors = open3d.Vector3dVector(colors)

    if T is not None:
        cloud.transform(T)
    return cloud

def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

def backproject_camera(im_depth, intrinsic_matrix, factor_depth):

    depth = im_depth.astype(np.float32, copy=True) / factor_depth

    # get intrinsic matrix
    K = np.matrix(intrinsic_matrix)
    K = np.reshape(K, (3,3))
    Kinv = np.linalg.inv(K)

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

    return np.array(X)

def get_4x4_transform(pose):
    object_pose_matrix4f = np.identity(4)
    object_pose = np.array(pose)
    if object_pose.shape == (4,4):
        object_pose_matrix4f = object_pose
    elif object_pose.shape == (3,4):
        object_pose_matrix4f[:3,:] = object_pose
    elif len(object_pose) == 7:
        object_pose_matrix4f[:3,:3] = quat2mat(object_pose[:4])
        object_pose_matrix4f[:3,-1] = object_pose[4:]
    else:
        print("[WARN]: Object pose for %s is not of shape (4,4) or (3,4) or 1-d quat (7)")
        return None
    return object_pose_matrix4f

def render_depth_pointcloud(im, depth, pose_data, points, intrinsic_matrix, factor_depth=1):

    rgb = im.copy()[:,:,::-1]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255

    X = backproject_camera(depth, intrinsic_matrix, factor_depth)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    coord_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

    if len(pose_data) == 0:
        open3d.draw_geometries([scene_cloud, coord_frame])
        return 

    all_objects_cloud = open3d.PointCloud()
    for pd in pose_data:
        object_cls = pd["cls"]
        object_pose = pd["pose"]
        # object_cloud_file = osp.join(object_model_dir,object_name,"points.xyz")
        object_pose_matrix4f = get_4x4_transform(object_pose)
        if object_pose_matrix4f is None:
            continue

        # object_pose_matrix4f[2,3]
        object_pts3d = points[object_cls] # read_xyz_file(object_cloud_file)
        object_cloud = create_cloud(object_pts3d)
        object_cloud.transform(object_pose_matrix4f)
        all_objects_cloud += object_cloud

        # print("Showing %s"%(object_name))
    open3d.draw_geometries([scene_cloud, all_objects_cloud, coord_frame])

def get_object_data(annotation, object_seg_ids):
    objects = annotation['objects']
    data = []
    for o in objects:
        d = {}
        q = o['quaternion_xyzw']
        t = np.array(o['location']) / 100  # cm to M
        pose = [q[-1],q[0],q[1],q[2],t[0],t[1],t[2]]
        d['pose'] = np.array(pose)
        d['transform'] = get_4x4_transform(pose)
        d['projected_cuboid'] = np.array(o['projected_cuboid'])
        d['visibility'] = o['visibility']
        cls = o['class']
        d['cls'] = cls
        d['seg_id'] = object_seg_ids[cls]
        data.append(d)
    return data

def get_object_poses(annotation):
    objects = annotation['objects']
    object_poses = []
    for o in objects:
        q = o['quaternion_xyzw']
        t = np.array(o['location']) / 100  # cm to M
        object_poses.append([q[-1],q[0],q[1],q[2],t[0],t[1],t[2]])
    return object_poses

def load_points_from_obj_file(obj_file):
    """Loads a Wavefront OBJ file. """
    vertices = []
    print("Loading obj file %s..."%(obj_file))

    material = None
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = values[1:4]
                # if swapyz:
                #     v = v[0], v[2], v[1]
                vertices.append(v)
    return np.array(vertices)

def download_models():
    """
    for i in "002_master_chef_can" "003_cracker_box" "004_sugar_box" "005_tomato_soup_can" "006_mustard_bottle" 
    "007_tuna_fish_can" "008_pudding_box" "009_gelatin_box" "010_potted_meat_can" "011_banana" "019_pitcher_base" "021_bleach_cleanser" 
    "024_bowl" "025_mug" "035_power_drill" "036_wood_block" "037_scissors" "040_large_marker" "051_large_clamp" 
    "052_extra_large_clamp" "061_foam_brick"; 
        do wget 'http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/'$i'_google_16k.tgz'; done
    """

def get_2d_projected_points(points, intrinsic_matrix, M):
    x3d = np.ones((4, len(points)), dtype=np.float32)
    x3d[0, :] = points[:,0]
    x3d[1, :] = points[:,1]
    x3d[2, :] = points[:,2]

    # projection
    RT = M[:3,:]
    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
    x = np.transpose(x2d[:2,:], [1,0]).astype(np.int32)
    return x

def visualize_pose(im, seg_mask, object_data, points, intrinsic_matrix):
    im_copy = im.copy()
    h,w = im_copy.shape[:2]
    mask = np.zeros((h,w,3), dtype=np.uint8)

    for pd in object_data:
        cls = pd["cls"]
        pose = pd["pose"]
        seg_id = pd['seg_id']
        vis = pd['visibility']
        if vis < 0.2:
            continue 

        color = get_random_color()
        cls_pts = points[cls]

        # projection
        M = get_4x4_transform(pose)
        if M is None:
            continue
        x = get_2d_projected_points(cls_pts, intrinsic_matrix, M)
        for px in x:
            px = tuple(px)
            cv2.circle(im_copy, px, 3, color, -1)
            if seg_mask[px[1],px[0]] == seg_id:
                cv2.circle(mask, px, 3, color, -1)

    cv2.imshow("proj_poses", im_copy)
    cv2.imshow("proj_poses_refined", mask)


def draw_cuboid_lines(img2, points, color):
    cv2.line(img2, points[0], points[1], color)
    cv2.line(img2, points[1], points[2], color)
    cv2.line(img2, points[3], points[2], color)
    cv2.line(img2, points[3], points[0], color)
    
    # draw back
    cv2.line(img2, points[4], points[5], color)
    cv2.line(img2, points[6], points[5], color)
    cv2.line(img2, points[6], points[7], color)
    cv2.line(img2, points[4], points[7], color)
    
    # draw sides
    cv2.line(img2, points[0], points[4], color)
    cv2.line(img2, points[7], points[3], color)
    cv2.line(img2, points[5], points[1], color)
    cv2.line(img2, points[2], points[6], color)

def visualize_proj_cuboid(im, object_data):
    im_copy = im.copy()
    h,w = im_copy.shape[:2]
    mask = np.zeros((h,w,3), dtype=np.uint8)
    for d in object_data:
        color = get_random_color()
        projc = d['projected_cuboid'].astype(np.int32)
        # cv2.fillConvexPoly(mask, projc[::2].copy(), color)

        projc = [tuple(pt) for pt in projc]
        for pt in projc:
            cv2.circle(im_copy, pt, 3, color, -1)
        draw_cuboid_lines(im_copy, projc, color)

    cv2.imshow("proj_cuboid", im_copy)
    # cv2.imshow("proj_cuboid_masks", mask)

coord_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

if __name__ == '__main__':
    import glob
    import json 

    ROOT_DIR = "/home/vincent/hd/datasets/FAT"
    MODEL_DIR = ROOT_DIR + "/models"
    SUPERCATEGORY = "FAT"

    # sample_dir = ROOT_DIR + "/single/002_master_chef_can_16k/temple_0"
    sample_dir = ROOT_DIR + "/mixed/temple_0"
    camera_type = "left"
    camera_settings_json = sample_dir + "/_camera_settings.json"
    object_settings_json = sample_dir + "/_object_settings.json"

    with open(camera_settings_json, "r") as f:
        camera_settings = json.load(f)
    with open(object_settings_json, "r") as f:
        object_settings = json.load(f)
    camera_settings = camera_settings['camera_settings']
    OBJ_CLASSES = object_settings["exported_object_classes"]
    # CLASSES = [c.lower().replace("_16k","") for c in OBJ_CLASSES]
    # cls_indexes = dict((k,ix) for ix, k in enumerate(OBJ_CLASSES))
    object_settings = object_settings["exported_objects"]
    object_seg_ids = dict((o['class'], o['segmentation_class_id']) for o in object_settings)
    for cs in camera_settings:
        if cs['name'] == camera_type:
            camera_settings = cs
            break

    # Convert transforms from column-major to row-major, and from cm to m
    object_transforms = dict((o['class'], np.array(o['fixed_model_transform']).transpose() / 100) for o in object_settings)
    intrinsic_matrix = get_camera_settings_intrinsic_matrix(camera_settings)

    points = {}
    for cls in OBJ_CLASSES:
        obj_file = MODEL_DIR + "/%s/google_16k/textured.obj"%(cls.lower().replace("_16k",""))
        obj_points = load_points_from_obj_file(obj_file)
        cloud = create_cloud(obj_points, T=object_transforms[cls])
        points[cls] = np.asarray(cloud.points)
        # open3d.draw_geometries([cloud, coord_frame])

    sample_file = sample_dir + "/000000.%s"%camera_type

    annot_file = sample_file + ".json"
    img_file = sample_file + ".jpg"
    seg_file = sample_file + ".seg.png"
    depth_file = sample_file + ".depth.png"


    with open(annot_file, "r") as f:
        annotation = json.load(f)

    object_data = get_object_data(annotation, object_seg_ids)

    # VISUALIZE

    factor_depth = 10000
    img = cv2.imread(img_file)
    img_height, img_width, _ = img.shape
    label = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    # render_depth_pointcloud(img, depth, object_data, points, intrinsic_matrix, factor_depth)
    visualize_pose(img, label, object_data, points, intrinsic_matrix)
    visualize_proj_cuboid(img, object_data)

    cv2.imshow("img", img)
    cv2.imshow("seg", label)
    cv2.waitKey(0)
