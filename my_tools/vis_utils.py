import numpy as np
from transforms3d.quaternions import quat2mat#, mat2quat

# extract vertmap for vertex predication
def extract_vertmap(labels, vertex_pred, num_classes):
    height = labels.shape[0]
    width = labels.shape[1]
    vertmap = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        I = np.where(labels == i)
        if len(I[0]) > 0:
            start = 3 * i
            end = 3 * i + 3
            vertmap[I[0], I[1], :] = vertex_pred[I[0], I[1], start:end]
    vertmap[:, :, 2] = np.exp(vertmap[:, :, 2])
    return vertmap

def vis_segmentations_vertmaps_detection(im, im_depth, im_labels, colors, center_map, 
        labels, rois, poses, intrinsic_matrix, num_classes, classes, points):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 3, 1)
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(3, 3, 2)
    plt.imshow(im_depth)
    ax.set_title('input depth')

    # show class label
    ax = fig.add_subplot(3, 3, 3)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    # if cfg.TEST.VERTEX_REG_2D:
    # show centers
    for i in xrange(rois.shape[0]):
        if rois[i, 1] == 0:
            continue
        cx = (rois[i, 2] + rois[i, 4]) / 2
        cy = (rois[i, 3] + rois[i, 5]) / 2
        w = rois[i, 4] - rois[i, 2]
        h = rois[i, 5] - rois[i, 3]
        if not np.isinf(cx) and not np.isinf(cy):
            plt.plot(cx, cy, 'yo')

            # show boxes
            plt.gca().add_patch(
                plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
                               edgecolor='g', linewidth=3))
        
    # show vertex map
    ax = fig.add_subplot(3, 3, 4)
    plt.imshow(center_map[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(3, 3, 5)
    plt.imshow(center_map[:,:,1])
    ax.set_title('centers y')
    
    ax = fig.add_subplot(3, 3, 6)
    plt.imshow(center_map[:,:,2])
    ax.set_title('centers z')

    # show projection of the poses
    # if cfg.TEST.POSE_REG:

    ax = fig.add_subplot(3, 3, 7, aspect='equal')
    plt.imshow(im)
    ax.invert_yaxis()
    for i in xrange(rois.shape[0]):
        cls = int(rois[i, 1])
        if cls > 0:
            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[i, :4])
            RT[:, 3] = poses[i, 4:7]
            print(classes[cls])
            print(RT)
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
            plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.5)
            # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

    ax.set_title('projection of model points')
    ax.invert_yaxis()
    ax.set_xlim([0, im.shape[1]])
    ax.set_ylim([im.shape[0], 0])

    # plt.ion()
    plt.show()
