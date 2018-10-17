import open3d
import numpy as np
import cv2

def read_xyz_file(file):
	with open(file, 'r') as f:
		d = np.array([l.strip("\n").split(" ") for l in f.readlines()], dtype=np.float32)
	return d

def get_depth_as_pointcloud(depth,fx,fy,cx,cy):
	rows, cols = depth.shape[:2]
	pts = np.zeros((rows,cols,3), dtype=np.float32)
	fx_i = 1/fx
	fy_i = 1/fy
	pts = []
	pts_indices = []
	for r in xrange(rows):
		for c in xrange(cols):
			depth_val = depth[r,c] 
			if depth_val > 0.001:
				x = (c + 0.5 - cx) * fx_i * depth_val
				y = (r + 0.5 - cy) * fy_i * depth_val
				z = depth_val
				# pts[r,c] = [x,y,z]
				pts.append([x,y,z])
				pts_indices.append([r,c])
	return pts, pts_indices

def get_rgb_indices(rgb,pts_indices):
	rgb_ = rgb[:,:,::-1]  # bgr to rgb
	rgb_ = rgb_.astype(np.float32) / 255  
	return [rgb_[pi[0],pi[1]] for pi in pts_indices]

rgb_file = "../data/demo_images/000005-color.png"
depth_file = "../data/demo_images/000005-depth.png"

rgb = cv2.imread(rgb_file)
depth  = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

camera_param_names = [ "fu", "fv", "u0", "v0", "k1", "k2", "k3" ]
camera_params = [ 1066.778, 1067.487, 312.9869, 241.3109, 0.04112172, -0.4798174, 1.890084 ]
fx,fy,cx,cy = camera_params[:4]
scale_factor = 10000
# min_d = 0.25
# max_d = 6
# max_min_diff = max_d - min_d

depth = depth.astype(np.float32) / scale_factor
# depth[depth!=0] += min_d

pts3d, pts_indices = get_depth_as_pointcloud(depth,fx,fy,cx,cy)
pts_rgb = get_rgb_indices(rgb,pts_indices)

scene_cloud = open3d.PointCloud(); 
scene_cloud.points = open3d.Vector3dVector(pts3d)
scene_cloud.colors = open3d.Vector3dVector(pts_rgb)

# open3d.draw_geometries([scene_cloud])
# cv2.imshow("rgb", rgb)
# cv2.imshow("depth", depth)
# cv2.waitKey(0)

pose_data = [
		{
			"name":"004_sugar_box", 
			"pose": np.array([[-0.05761516,-0.15287192,0.98656505,-0.02269021],[-0.8910858,0.45346868,0.01822745,0.10536116],[-0.45016283,-0.878064,-0.16234866,0.8350466]])
		},
		{
			"name":"021_bleach_cleanser", 
			"pose": np.array([[-0.90805304,-0.4185382,-0.01629682,-0.12704515],[-0.22480434,0.5198223,-0.8241649,-0.04170364],[0.35341594,-0.74472183,-0.5661155,0.9612876]])	
		},
		{
			"name":"035_power_drill", 
			"pose": np.array([[-0.79307294,  0.03152161,  0.6083107,   0.01751409],
							 [-0.28632876, -0.9007414,  -0.32662052, -0.01525172],
							 [ 0.53763497, -0.4332107,   0.723379,    0.88914335]])
		},
		{
			"name":"003_cracker_box", 
			"pose": np.array([[-0.23085807,0.9696965,0.0799573,-0.10856971],[0.4966099,0.18809806,-0.84734744,-0.08332705],[-0.8367097,-0.15590942,-0.52498484,1.0072432]])	
		},
		{
			"name":"051_large_clamp", 
			"pose": np.array([[0.9829423,-0.18388733,0.0031483,0.09879504],[-0.09907233,-0.5438435,-0.8333181,0.07760569],[0.15494882,0.8187917,-0.5527849,0.96678644]])	
		}
]

object_model_dir = "../data/LOV/models/"
for pd in pose_data:
	object_name = pd["name"]
	object_pose = pd["pose"]
	object_cloud_file = object_model_dir + object_name + "/points.xyz"
	object_pose_matrix4f = np.vstack((object_pose, np.zeros(4)))
	object_pose_matrix4f[-1,-1] = 1
	# object_pose_T = object_pose[:,3]
	# object_pose_R = object_pose[:,:3]

	object_pts3d = read_xyz_file(object_cloud_file)
	object_cloud = open3d.PointCloud(); 
	object_cloud.points = open3d.Vector3dVector(object_pts3d)
	object_cloud.transform(object_pose_matrix4f)

	open3d.draw_geometries([scene_cloud, object_cloud])

