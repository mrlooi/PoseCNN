import numpy as np
import cv2
from transforms3d.quaternions import mat2quat, quat2mat


def get_camera_settings_intrinsic_matrix(camera_settings):
    intrinsic_settings = camera_settings['intrinsic_settings']
    intrinsic_matrix = np.identity(3)
    intrinsic_matrix[0,0] = intrinsic_settings['fx']
    intrinsic_matrix[1,1] = intrinsic_settings['fy']
    intrinsic_matrix[0,2] = intrinsic_settings['cx']
    intrinsic_matrix[1,2] = intrinsic_settings['cy']
    return intrinsic_matrix

if __name__ == '__main__':
    import glob
    import json 

    ROOT_DIR = "/home/vincent/hd/datasets/FAT"
    MODEL_DIR = ROOT_DIR + "/models"
    SUPERCATEGORY = "FAT"

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
    CLASSES = [c.lower().replace("_16k","") for c in OBJ_CLASSES]
    print(CLASSES)
    object_settings = object_settings["exported_objects"]
    for cs in camera_settings:
        if cs['name'] == camera_type:
            camera_settings = cs
            break

    points = [[]] # dummy list for background class
    for cls in CLASSES[1:]:
        point_file = MODEL_DIR + "/%s/points.xyz"%(cls)
        points.append(np.loadtxt(point_file))

    intrinsic_matrix = get_camera_settings_intrinsic_matrix(camera_settings)


    sample_file = sample_dir + "/000000.%s"%camera_type

    annot_file = sample_file + ".json"
    img_file = sample_file + ".jpg"
    seg_file = sample_file + ".seg.png"

    img = cv2.imread(img_file)
    img_height, img_width, _ = img.shape
    label = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)

    with open(annot_file, "r") as f:
        annotation = json.load(f)

    cv2.imshow("img", img)
    cv2.waitKey(0)
