import open3d
import numpy as np
import cv2
import scipy.io as sio
from transforms3d.quaternions import quat2mat, mat2quat

global cnt
cnt = 0
def visualize(im, depth, label, centers):
    global cnt
    cnt += 1
    h,w = label.shape
    label_m = np.zeros((h,w,3), dtype=np.uint8)
    for cls in np.unique(label):
        label_m[label==cls] = np.random.randint(0,255,size=3)
    for c in centers:
        cv2.circle(im, tuple(c.astype(np.int32)), 3, (0,255,0), -1)
    cv2.imshow('im%d'%(cnt), im)
    cv2.imshow('depth', depth)
    cv2.imshow('label', label_m)

def visualize_pose(im, cls_indexes, poses, points, intrinsic_matrix):
    im_copy = im.copy()
    for ix,cls in enumerate(cls_indexes):
        color = np.random.randint(0,255,size=(3))
        cls_pts = points[cls]
        x3d = np.ones((4, len(cls_pts)), dtype=np.float32)
        x3d[0, :] = cls_pts[:,0]
        x3d[1, :] = cls_pts[:,1]
        x3d[2, :] = cls_pts[:,2]

        # projection
        RT = np.zeros((3, 4), dtype=np.float32)
        pose = poses[ix]
        RT[:,:3] = pose[:, :3]
        RT[:,3] = pose[:, 3]
        print(RT)
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        x = np.transpose(x2d[:2,:], [1,0]).astype(np.int32)
        for px in x:
            # im_copy[px[1],px[0]] = color
            cv2.circle(im_copy, tuple(px), 3, color, -1)
        # plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.5)

    cv2.imshow("proj_poses%d"%(cnt), im_copy)

def mirror_pose_along_y_axis(pose):
    R = pose[:, :3]
    # q = mat2quat(R)
    # w,x,y,z = q
    # q = [w,x,-y,-z]
    # q = [-x,y,z,-w]
    # q[2] *= -1
    # pose[:, :3] = quat2mat(q)
    M_x_axis = np.identity(4)
    M_x_axis[1,1] = -1
    M_x_axis[0,0] = -1
    pose = np.dot(pose, M_x_axis)
    pose[0, 3] *= -1
    return pose

if __name__ == '__main__':
    _classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    
    root_dir = "./data/LOV/" 
    points = [[]] # dummy list for background class
    for cls in _classes[1:]:
        point_file = root_dir + "models/%s/points.xyz"%(cls)
        points.append(np.loadtxt(point_file))
        
    file_names = ["0000/000001", "0001/000001", "0002/000001"]

    data_dir = root_dir + "data_orig/"
    data_dir2 = root_dir + "data/"
    for f in file_names:
        base_f = data_dir + f
        im_file = base_f + "-color.png"
        depth_file = base_f + "-depth.png"
        label_file = base_f + "-label.png"
        meta_file = base_f + "-meta.mat"

        meta = sio.loadmat(meta_file)
        intrinsic_matrix = meta['intrinsic_matrix']
        center = meta['center']

        im = cv2.imread(im_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        h,w,_ = im.shape

        # RESIZE
        im = cv2.resize(im, (w/2,h/2))
        depth = cv2.resize(depth, (w/2,h/2))
        label = cv2.resize(label, (w/2,h/2))
        vertmap = meta['vertmap']
        vertmap = cv2.resize(vertmap, (w/2,h/2))
        intrinsic_matrix[0,0] /= 2
        intrinsic_matrix[0,2] /= 2
        intrinsic_matrix[1,1] /= 2
        intrinsic_matrix[1,2] /= 2
        center /= 2
        meta['intrinsic_matrix'] = intrinsic_matrix
        meta['center'] = center
        meta['vertmap'] = vertmap
        new_im_file = im_file.replace(data_dir, data_dir2)
        new_depth_file = depth_file.replace(data_dir, data_dir2)
        new_label_file = label_file.replace(data_dir, data_dir2)
        new_meta_file = meta_file.replace(data_dir, data_dir2)
        cv2.imwrite(new_im_file, im)
        cv2.imwrite(new_depth_file, depth)
        cv2.imwrite(new_label_file, label)
        sio.savemat(new_meta_file, meta)
        print("Saved to %s, %s, %s, %s"%(new_im_file, new_depth_file, new_label_file, new_meta_file))
        
        cls_indexes = meta['cls_indexes'].squeeze()
        poses = meta['poses']
        poses = [poses[:,:,ix] for ix in xrange(len(cls_indexes))]

        visualize(im, depth, label, center)
        visualize_pose(im, cls_indexes, poses, points, intrinsic_matrix)
        cv2.waitKey(0)

        # flipped = 1
        # if flipped:
        #     im = im[:,::-1,:]
        #     depth = depth[:,::-1]
        #     center[:,0] = w - center[:,0] + 1  # horizontal flip only
        #     im = im.astype(np.uint8).copy()

        #     for ix in xrange(len(poses)):
        #         poses[ix] = mirror_pose_along_y_axis(poses[ix])

        #     visualize(im, depth, center)
        #     visualize_pose(im, cls_indexes, poses, points, intrinsic_matrix)
        #     cv2.waitKey(0)
