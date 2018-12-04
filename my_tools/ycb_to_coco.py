import numpy as np
import cv2
import json
import os.path as osp
import copy
import datetime
from transforms3d.quaternions import mat2quat


def convert_datetime_to_string(dt=datetime.datetime.now(), formt="%Y-%m-%d %H:%M:%S"):
    return dt.strftime(formt)

class CocoAnnotationClass(object):
    def __init__(self, classes, supercategory=""):
        self.classes = classes
        self.map_classes_idx = {c: ix+1 for ix,c in enumerate(classes)}  # coco is 1-indexed
        self.map_idx_classes = {v:k for k,v in self.map_classes_idx.items()}
        self.data = self._get_default_data()
        for c,idx in self.map_classes_idx.items():
            self._add_category(c,idx,supercategory)

    def _get_default_data(self):
        default_d = {
            "info": {
                "year" : 2018, 
                "version" : "", 
                "description" : "", 
                "contributor" : "", 
                "url" : "", 
                "date_created" : convert_datetime_to_string()
            },
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [
                {
                    "id" : 1, 
                    "name" : "", 
                    "url" : ""
                }
            ]
        }
        return default_d

    def set_classes(self, classes):
        self.classes = classes

    def clear(self):
        self.data = self._get_default_data()

    def _add_category(self, name, id=None, supercategory=""):
        cat_id = len(self.data["categories"]) + 1 if id is None else id
        cat_data = {
                    "id" : cat_id, 
                    "name" : name, 
                    "supercategory" : supercategory
                }
        self.data["categories"].append(cat_data)

    def add_annot(self, id, img_id, img_cls, seg_data, meta_data={}, is_crowd=0):
        """ONLY SUPPORTS seg polygons of len 1 i.e. cannot support multiple polygons that refer to the same id"""
        if isinstance(img_cls, str):
            if img_cls not in self.map_classes_idx:
                print("%s not in coco classes!"%(img_cls))
                return 
            cat_id = self.map_classes_idx[img_cls]
        else:
            assert img_cls in self.map_idx_classes
            cat_id = img_cls
        seg_data_arr = np.array(seg_data)
        assert(len(seg_data_arr.shape) == 2)
        bbox = np.array([np.amin(seg_data_arr, axis=0), np.amax(seg_data_arr, axis=0)]).reshape(4)
        bbox[2:] -= bbox[:2]
        bbox = bbox.tolist()
        area = cv2.contourArea(seg_data_arr)
        annot_data =    {
                    "id" : id,
                    "image_id" : img_id,
                    "category_id" : cat_id,
                    "segmentation" : [seg_data_arr.flatten().tolist()],
                    "area" : area,
                    "bbox" : bbox,
                    "iscrowd" : is_crowd,
                    "meta": meta_data # CUSTOM
                }
        self.data["annotations"].append(annot_data)

    def add_image(self, id, width, height, file_name, date_captured=convert_datetime_to_string()):
        img_data =  {
                    "id" : id,
                    "width" : width,
                    "height" : height,
                    "file_name" : file_name,
                    "license" : 1,
                    "flickr_url" : "",
                    "coco_url" : "",
                    "date_captured" : date_captured
                }
        self.data["images"].append(img_data)

    def get_annot_json(self):
        return copy.deepcopy(self.data)


def get_cls_contours(label, cls):
    mask = np.zeros((label.shape), dtype=np.uint8)
    y,x = np.where(label==cls)
    mask[y,x] = 255

    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    return contours

def approx_contour(cnt, eps=0.005):
    if len(cnt) == 0:
        return []
    arclen = cv2.arcLength(cnt, True)
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = approx.squeeze()
    return approx

def convert_contours_to_polygons(contours, eps=0.005):
    polygons = [approx_contour(cnt, eps) for cnt in contours]
    polygons = [p for p in polygons if len(p) >= 3] # need at least 3 points to form a polygon
    return polygons

def vote_vertex_centers(im_label, cls_indexes, center, poses):
    width = im_label.shape[1]
    height = im_label.shape[0]
    cls_i = cls_indexes.squeeze()

    vertex_targets = np.zeros((len(cls_i), height, width, 3), dtype=np.float32)
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
        vertex_targets[ind, y, x, 1] = R[1,:]
        vertex_targets[ind, y, x, 0] = R[0,:]
        vertex_targets[ind, y, x, 2] = z

    return vertex_targets

if __name__ == '__main__':
    import glob
    import scipy.io as sio

    SUPERCATEGORY = "YCB"
    CLASSES = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)

    coco_out_file = "./coco_lov_debug.json"
    coco_annot = CocoAnnotationClass(CLASSES[1:], SUPERCATEGORY) # COCO IS 1-indexed, don't include BG CLASS

    root_dir = "./data/LOV/" 
    points = [[]] # dummy list for background class
    for cls in CLASSES[1:]:
        point_file = root_dir + "models/%s/points.xyz"%(cls)
        points.append(np.loadtxt(point_file))
        
    file_names = ["0000/000001", "0001/000001", "0002/000001"]

    data_dir = root_dir + "data/"
    
    total_cnt = 0

    VIS = True

    for fx,f in enumerate(file_names):
        base_f = data_dir + f
        im_file = base_f + "-color.png"
        # depth_file = base_f + "-depth.png"
        label_file = base_f + "-label.png"
        meta_file = base_f + "-meta.mat"

        # load img and gt mask labels
        img = cv2.imread(im_file)
        img_height, img_width, _ = img.shape
        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        assert label.shape == (img_height, img_width)

        # load meta data
        meta = sio.loadmat(meta_file)
        intrinsic_matrix = meta['intrinsic_matrix']
        cls_indexes = meta['cls_indexes'].squeeze().astype(np.int32)
        center = np.round(meta['center']).astype(np.int32)
        poses = meta['poses'].astype(np.float32)
        poses = [poses[:,:,ix] for ix in xrange(len(cls_indexes))]  # 3x4 matrix
        quats = [mat2quat(p[:,:-1]) for p in poses]
        poses = [np.hstack((quats[ix],p[:,-1])) for ix,p in enumerate(poses)]

        IMG_ID = fx + 1

        cnt = 0 

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        for cx,cls in enumerate(cls_indexes):

            # convert labels to polygons
            cls = int(cls)
            cls_contours = get_cls_contours(label, cls)
            polygons = convert_contours_to_polygons(cls_contours, eps=0.003)

            if len(polygons) == 0:
                continue

            meta_data = {'center': center[cx].tolist(), 'pose': poses[cx].flatten().tolist(), 'intrinsic_matrix': intrinsic_matrix.tolist()}

            cnt += 1
            total_cnt += 1

            coco_annot.add_annot(total_cnt, IMG_ID, cls, polygons[0], meta_data)

            if VIS:
                approx = polygons[0]
                total = len(approx)
                img_copy = img.copy()
                for j,p in enumerate(approx):
                    cv2.circle(img_copy, tuple(p), 5, BLUE, -1)
                    cv2.line(img_copy, tuple(p), tuple(approx[(j+1)%total]), BLUE, 3)
                # img_copy = cv2.fillPoly(img_copy, [approx], GREEN)
                img_copy = cv2.putText(img_copy, CLASSES[cls], tuple(np.mean(approx, axis=0).astype(np.int32)), cv2.FONT_HERSHEY_COMPLEX, 1.0, BLUE)
                cv2.imshow("img", img_copy)
                cv2.waitKey(0)

        if cnt == 0:
            continue

        img_name = im_file.replace(data_dir, "")
        coco_annot.add_image(IMG_ID, img_width, img_height, img_name)

    with open(coco_out_file, "w") as f:
        json.dump(coco_annot.get_annot_json(), f)
        print("Saved to %s"%(coco_out_file))

