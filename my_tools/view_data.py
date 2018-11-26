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

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

def visualize_vertmap(vertmap):
    cx = normalize(vertmap[:,:,0])
    cy = normalize(vertmap[:,:,1])
    cz = normalize(vertmap[:,:,2])
    cv2.imshow("vertmap x", cx)
    cv2.imshow("vertmap y", cy)
    cv2.imshow("vertmap z", cz)

def visualize_centers(im_label, cls_indexes, center, poses):
    width = im_label.shape[1]
    height = im_label.shape[0]
    cls_i = cls_indexes.squeeze()

    # vertex_targets = np.zeros((len(cls_i), height, width, 3), dtype=np.float32)
    vertex_targets = np.zeros((height, width, 3), dtype=np.float32)
    # vertex_weights = np.zeros(vertex_targets.shape, dtype=np.float32)

    c = np.zeros((2, 1), dtype=np.float32)
    for ind, cls in enumerate(cls_i):
        c[0] = center[ind, 0]
        c[1] = center[ind, 1]
        z = poses[ind][2, 3]

        y, x = np.where(im_label == cls)

        R = c - np.vstack((x, y))
        # compute the norm
        N = np.linalg.norm(R, axis=0) + 1e-10
        # normalization
        R = R / N # np.divide(R, np.tile(N, (2,1)))
        # assignment
        vertex_targets[y, x, 0] = R[0,:]
        vertex_targets[y, x, 1] = R[1,:]
        vertex_targets[y, x, 2] = z

        # vertex_targets[ind, y, x, 0] = R[0,:]
        # vertex_targets[ind, y, x, 1] = R[1,:]
        # vertex_targets[ind, y, x, 2] = z
        # vertex_weights[ind, y, x, :] = 10.0

    min_depth = 0
    max_depth = 10
    cx = normalize(vertex_targets[:,:,0],-1,1)
    cy = normalize(vertex_targets[:,:,1],-1,1)
    cz = normalize(vertex_targets[:,:,2], min_depth)#, max_depth)
    cv2.imshow("center x", cx)
    cv2.imshow("center y", cy)
    cv2.imshow("center z", cz)
    return vertex_targets#, vertex_weights

# def visualize_centers(vert_centers):
#     min_depth = 0
#     max_depth = 10

#     if len(vert_centers) == 0:
#         return
#     centers_mat = []
#     for ix,vc in enumerate(vert_centers):
#         # color = np.random.randint(0,255,size=(3))
#         # np.where()
#         cx = normalize(vc[:,:,0],-1,1)
#         cy = normalize(vc[:,:,1],-1,1)
#         cz = normalize(vc[:,:,2], min_depth, max_depth)
#         merged = np.hstack((cx,cy,cz))
#         if ix == 0:
#             centers_mat = merged
#         else:
#             centers_mat = np.vstack((centers_mat, merged))
#         # cv2.imshow("centers y", cy)
#         # cv2.imshow("centers z", cz)
#     cm_resized = cv2.resize(centers_mat, (1333, 1000))
#     cv2.imshow("centers_mat", cm_resized)

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
        vertmap = meta['vertmap']
        h,w,_ = im.shape

        # # RESIZE
        # im = cv2.resize(im, (w/2,h/2))
        # depth = cv2.resize(depth, (w/2,h/2))
        # label = cv2.resize(label, (w/2,h/2))
        # vertmap = cv2.resize(vertmap, (w/2,h/2))
        # intrinsic_matrix[0,0] /= 2
        # intrinsic_matrix[0,2] /= 2
        # intrinsic_matrix[1,1] /= 2
        # intrinsic_matrix[1,2] /= 2
        # center /= 2
        # meta['intrinsic_matrix'] = intrinsic_matrix
        # meta['center'] = center
        # meta['vertmap'] = vertmap
        # new_im_file = im_file.replace(data_dir, data_dir2)
        # new_depth_file = depth_file.replace(data_dir, data_dir2)
        # new_label_file = label_file.replace(data_dir, data_dir2)
        # new_meta_file = meta_file.replace(data_dir, data_dir2)
        # cv2.imwrite(new_im_file, im)
        # cv2.imwrite(new_depth_file, depth)
        # cv2.imwrite(new_label_file, label)
        # sio.savemat(new_meta_file, meta)
        # print("Saved to %s, %s, %s, %s"%(new_im_file, new_depth_file, new_label_file, new_meta_file))
        
        cls_indexes = meta['cls_indexes'].squeeze()
        poses = meta['poses']
        poses = [poses[:,:,ix] for ix in xrange(len(cls_indexes))]


        visualize(im, depth, label, center)
        visualize_pose(im, cls_indexes, poses, points, intrinsic_matrix)
        vert_centers = visualize_centers(label, cls_indexes, center, poses)
        # visualize_vertmap(vertmap)
        # visualize_centers(vert_centers)
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
