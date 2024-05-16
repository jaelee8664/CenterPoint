import argparse
import json
import os
import pickle
import sys
import time
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
import torch
import numpy as np
from pathlib import Path
import yaml
from det3d.core import box_np_ops
import matplotlib.pyplot as plt
import cv2
import PIL
import open3d as o3d
from det3d.torchie import Config
from yoloX.yolox_utils import visual
import open3d as o3d
import copy
torch.multiprocessing.set_sharing_strategy('file_system')
def z_offset(data_batch, val):
    for points in data_batch["points"]:
        points[:,2] += val
    data_batch["voxels"][..., 2] += val

def pnt3d_to_img(lidar_states, calib, offset, cam):
	lidar_states = np.concatenate((lidar_states, np.ones(1)))
	lidar_states = lidar_states.reshape((lidar_states.shape[0], 1))
	lidar_states = np.matmul(cam, np.matmul(np.matmul(calib, offset), lidar_states).T[:, :3].T).T
	
	for k in range(lidar_states.shape[0]):
		lidar_states[k, 0] = lidar_states[k, 0] / lidar_states[k, 2]
		lidar_states[k, 1] = lidar_states[k, 1] / lidar_states[k, 2]
		lidar_states[k, 2] = 1
	return lidar_states # same as cam_states

def pnt2d_img_to_lidar(img_states, depth_img, inv_cam, inv_camlib4, test_size):
    """
    Args:
        img_states (np.array): 2d 박스 x1y1x2y2 상태와 score, class 정보를 담은 N x 7 array
    Returns:
        2d box 가운데의 lidar 좌표계에서의 위치인 N x 3 array
    """
    lidar_states = np.ones((4, img_states.shape[0]))
    img_states = img_states.cpu().numpy()
    bboxes = img_states[:, 0:4]

    # preprocessing: resize
    scale = min(
                test_size[0] / float(depth_img.shape[0]), test_size[1] / float(depth_img.shape[1])
            )
    bboxes /= scale
    x_states = (bboxes[:, 0] + bboxes[:, 2]) // 2
    y_states = (bboxes[:, 1] + bboxes[:, 3]) // 2
    lidar_states[0, :] = x_states.T
    lidar_states[1, :] = y_states.T
    for i in range(img_states.shape[0]):
        if 0 <= x_states[i] < depth_img.shape[1] and 0 <= y_states[i] < depth_img.shape[0]:
            z = depth_img[int(y_states[i]), int(x_states[i])]
            lidar_states[:3, i] *= z
        else:
            lidar_states[:3, i] = 0
    lidar_states[:3, :] = np.matmul(inv_cam, lidar_states[:3, :])
    lidar_states = np.matmul(inv_camlib4, lidar_states)
    return lidar_states


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", default="configs/onlypoints/pp/kitech_centerpoint_pp.py", help="train config file path")
    parser.add_argument("--work_dir", default = "work_dirs/kitech_data", help="directory of 3D projected images")
    # parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--load_from", default = "kitech_inputoutput_result.pickle", type=str, help="load saved pickle file")
    parser.add_argument("--save_to", type=str, default="final_visualize_partial", help="save directory")
    parser.add_argument("--z_offset", type=float, default=0.0, help="z offset")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    # parser.add_argument(
    #     "--launcher",
    #     choices=["pytorch", "slurm"],
    #     default="pytorch",
    #     help="job launcher",
    # )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args
args = parse_args()
cfg = Config.fromfile(args.config)
# distribution 설정 안함
cfg.local_rank = args.local_rank 

# init logger before other steps
distributed = False
cfg.gpus = args.gpus

if os.path.exists(args.load_from):
    with open(args.load_from, "rb") as f:
        input_output = pickle.load(f)
else:
    raise NotImplementedError

line_list = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 0],
             [4, 5],
             [5, 6],
             [6, 7],
             [7, 4],
             [0, 4],
             [1, 5],
             [2, 6],
             [3, 7]]

# calib = np.array([[-0.99983949, -0.01008867, -0.01480574, 0.05565861],
# [0.01433594, 0.04514753, -0.99887746, 0.08249057],
# # [0.01074579, -0.99892939, -0.04499565, -0.00898347],
# [0.01074579, -0.99892939, -0.04499565, 0.8-0.00898347],
# [0, 0, 0, 1]])
# offset = [[1,0,0,0], [0,1,0,0], [0,0,1,0.8],[0,0,0,1]]
# # calib = np.linalg.inv(calib)
# cam = np.array([[1076.7392578125, 0.0, 957.8059692382812],
# [0.0, 1076.7392578125, 546.310791015625],
# [0.0, 0.0, 1.0]])
# # cam = [[1072.7210693359375, 0.0, 980.5731201171875],
# # [0.0, 1072.7210693359375, 554.7196044921875],
# # [0.0, 0.0, 1.0]]

cam = np.zeros([3,3])
cam[0,0] = 262.49102783203125
cam[1,1] = 262.49102783203125
cam[2,2] = 1
cam[0,2] = 327.126708984375
cam[1,2] = 184.74203491210938
calib = np.array([[ 0, -1,  0,  0.02],
                  [ 0,  0, -1, -0.17],
                  [ 1,  0,  0, -0.06]])
offset = np.identity(4)
calib4 = np.concatenate((calib, np.array([[0,0,0,1]])), axis=0)
inv_camlib4 = np.linalg.inv(calib4)
inv_cam = np.linalg.inv(cam)
offset = np.identity(4)
test_size = (800, 1440)
os.makedirs(args.save_to, exist_ok=True)

def box3d_projection(img):
	boxes3d_pts_list = []
	# pred_boxes 그리기
	for idx in range(len(pred_boxes)):
		if scores[idx] < 0.5:
			continue
		if classes[idx] != 8:
			continue
		translation = pred_boxes[idx][:3]
		if translation[0] <= 0.1: # 카메라 뒤쪽 상황은 제외
			continue
		pnt2d = pnt3d_to_img(translation, calib, offset, cam)[0]
		if pnt2d[0] < 0 or pnt2d[0] >= 640 or pnt2d[1] < 0 or pnt2d[1] >= 360:
			continue
		w, l, h = pred_boxes[idx][3], pred_boxes[idx][4], pred_boxes[idx][5]
		rotation = pred_boxes[idx][-1]

		bounding_box = np.array([
						[-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
						[w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
						[-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]]) 
		rotation_matrix = np.array([
				[np.cos(rotation), -np.sin(rotation), 0.0],
				[np.sin(rotation), np.cos(rotation), 0.0],
				[0.0, 0.0, 1.0]])
		eight_points = np.tile(translation, (8, 1))

		corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
		boxes3d_pts = corner_box.transpose()
		boxes3d_pts[:, 2] = np.clip(boxes3d_pts[:, 2], -1.5, 60) # 박스의 아랫부분을 잘라내기 위한 클리핑
		boxes3d_pts = boxes3d_pts.T
		
		boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts.T)
		boxes3d_pts_list.append(np.asarray(boxes3d_pts))

	for j in range(len(boxes3d_pts_list)):
		# box_pt: (8, 3)
		boxes3d_pts_list[j] = np.concatenate((boxes3d_pts_list[j], np.ones((boxes3d_pts_list[j].shape[0], 1))), axis=1)
		boxes3d_pts_list[j] = np.matmul(cam, np.matmul(np.matmul(calib, offset), boxes3d_pts_list[j].T).T[:, :3].T).T
		
		for k in range(boxes3d_pts_list[j].shape[0]):
			boxes3d_pts_list[j][k, 0] = boxes3d_pts_list[j][k, 0] / boxes3d_pts_list[j][k, 2]
			boxes3d_pts_list[j][k, 1] = boxes3d_pts_list[j][k, 1] / boxes3d_pts_list[j][k, 2]
			boxes3d_pts_list[j][k, 2] = 1

		for l in range(len(line_list)):
			pnts = line_list[l]
			cv2.line(img, (boxes3d_pts_list[j][pnts[0], 0].astype(np.int32), boxes3d_pts_list[j][pnts[0], 1].astype(np.int32)), (boxes3d_pts_list[j][pnts[1], 0].astype(np.int32), boxes3d_pts_list[j][pnts[1], 1].astype(np.int32)), (255, 0, 0), thickness = 2)
			# cv2.line(img4, (boxes3d_pts_list[j][pnts[0], 0].astype(np.int32), boxes3d_pts_list[j][pnts[0], 1].astype(np.int32)), (boxes3d_pts_list[j][pnts[1], 0].astype(np.int32), boxes3d_pts_list[j][pnts[1], 1].astype(np.int32)), (255, 0, 0), thickness = 5)
	return img

def pnts_in_box_projection(img, dist_thres=0.1, score_thres=0.5):
	# Create boolean masks for each condition
	mask1 = pred_boxes[..., 0] > dist_thres
	mask2 = scores >= score_thres
	mask3 = input_output["output_classes"][i] == 8
	# Combine all masks using logical AND operation
	intersection_mask = np.logical_and(np.logical_and(mask1, mask2), mask3)
	if pred_boxes[intersection_mask].shape[0]:
		point_indices = box_np_ops.points_in_rbbox(points_v, pred_boxes[intersection_mask])
		mask_boxes = np.zeros((point_indices.shape[0]))
		for j in range(point_indices.shape[1]):
			mask_boxes = np.logical_or(mask_boxes, point_indices[:, j])
		inboxpnts = points_v[ mask_boxes]
		points_v_appended =  np.concatenate((inboxpnts, np.ones((inboxpnts.shape[0], 1))), axis=1)
		# points_v_appended =  np.concatenate((points_v, np.ones((points_v.shape[0], 1))), axis=1)
		cam_points = np.matmul(cam, np.matmul(np.matmul(calib, offset), points_v_appended.T).T[:, :3].T).T
		for k in range(cam_points.shape[0]):
			cam_points[k, 0] = cam_points[k, 0] / cam_points[k, 2]
			cam_points[k, 1] = cam_points[k, 1] / cam_points[k, 2]
			cam_points[k, 2] = 1
		for l in range(cam_points.shape[0]):
			pnts = cam_points[l, :]
			if 0 <= pnts[0].astype(np.int32) < 640 and 0 <= pnts[1].astype(np.int32) < 360:
				cv2.line(img, (pnts[0].astype(np.int32), pnts[1].astype(np.int32)), (pnts[0].astype(np.int32), pnts[1].astype(np.int32)), (0, 0, 255), thickness = 2)
	return img

def box2d_visualize(img, score_conf=0.5):
	if pred_boxes_2d is not None:
		result_img_rgb, bboxes, scores, cls, cls_names = visual(pred_boxes_2d, img, test_size, cls_conf=score_conf)
	else:
		result_img_rgb = img
	return result_img_rgb

partial_sequence = [[1180, 1440], [3750, 3825], [4420, 4470], [4650, 4770], [4840, 4963], [7845, 8135]]

# for seq in partial_sequence:
# 	for i in tqdm(range(seq[0], seq[1])):
# 		points_v = input_output["points"][i]
# 		pred_boxes = input_output["output_boxes"][i]
# 		pred_boxes_2d = input_output["2D_states"][i]
# 		scores = input_output["output_scores"][i]
# 		classes = input_output["output_classes"][i]
# 		mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
# 		pcd = o3d.geometry.PointCloud()
# 		pcd.points = o3d.utility.Vector3dVector(points_v)
# 		rgb_name = Path(str(input_output["file_name"][i].stem) + "_rgb.jpg")
# 		depth_name = Path(str(input_output["file_name"][i].stem) + "_d.png")
# 		img3d_name = Path(input_output["file_name"][i]).with_suffix('.png')
# 		rgb_file = Path(cfg.data_root)/Path("rgbd")/rgb_name
# 		depth_file = Path(cfg.data_root)/Path("rgbd")/depth_name
# 		img3d_file = Path(args.work_dir)/img3d_name
# 		img = np.array(PIL.Image.open(rgb_file), dtype=np.uint8)
# 		depth_raw = np.array(PIL.Image.open(depth_file), dtype=np.uint16)
# 		depth = depth_raw / 200
# 		img3d = np.array(PIL.Image.open(img3d_file), dtype=np.uint8).reshape((720, 1280, 3))

# 		# # 3d 박스 이미지 좌표계에 투영
# 		# img2 = box3d_projection(copy.deepcopy(img))
# 		# # 3d 박스안에 있는 포인트만 이미지 좌표계에 투영
# 		# img3 = pnts_in_box_projection(copy.deepcopy(img2), dist_thres=0.1, score_thres=0.5)
# 		# # 2d 박스 이미지에 플롯
# 		# img4 = box2d_visualize(copy.deepcopy(img2))

# 		fig = plt.figure(figsize=(20,20))
# 		# ax1 = fig.add_subplot(2, 2, 1)
# 		# ax1.imshow(img)
# 		# ax1.set_title('raw image')
# 		# ax1.axis("off")
# 		"""
# 		If it's needed to make a subplot of a custom size, one way is to pass (left, bottom, width, height) information to a add_axes() call on the figure object.
# 		"""
# 		# ax2 = fig.add_subplot(2, 1, 1)
# 		ax2 = fig.add_axes([0.05, 0.31, 0.9, 0.8])
# 		ax2.imshow(img3d)
# 		ax2.set_title('rgbd points+ lidar points + 3D boxes')
# 		ax2.axis("off")

# 		# ax3 = fig.add_subplot(2, 2, 3)
# 		ax3 = fig.add_axes([0.01, 0, 0.45, 0.3])
# 		ax3.imshow(img)
# 		ax3.set_title('color image')
# 		# ax3.imshow(img3)
# 		# ax3.set_title('projected points and boxes')
# 		ax3.axis("off")

# 		# ax4 = fig.add_subplot(2, 2, 4)
# 		ax4 = fig.add_axes([0.54, 0, 0.45, 0.3])
# 		ax4.imshow(depth_raw)
# 		ax4.set_title('depth image')
# 		# ax4.imshow(img4)
# 		# ax4.set_title('2D boxes and projected boxes')
# 		ax4.axis("off")

# 		fig.savefig(args.save_to + "/" + str(input_output["file_name"][i].with_suffix('.png')))
# 		plt.cla()
# 		plt.clf()
# 		plt.close(fig)


for i in tqdm(range(len(input_output["points"]))):
	points_v = input_output["points"][i]
	pred_boxes = input_output["output_boxes"][i]
	pred_boxes_2d = input_output["2D_states"][i]
	scores = input_output["output_scores"][i]
	classes = input_output["output_classes"][i]
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points_v)
	rgb_name = Path(str(input_output["file_name"][i].stem) + "_rgb.jpg")
	depth_name = Path(str(input_output["file_name"][i].stem) + "_d.png")
	img3d_name = Path(input_output["file_name"][i]).with_suffix('.png')
	rgb_file = Path(cfg.data_root)/Path("rgbd")/rgb_name
	depth_file = Path(cfg.data_root)/Path("rgbd")/depth_name
	img3d_file = Path(args.work_dir)/img3d_name
	img = np.array(PIL.Image.open(rgb_file), dtype=np.uint8)
	depth_raw = np.array(PIL.Image.open(depth_file), dtype=np.uint16)
	depth = depth_raw / 200
	img3d = np.array(PIL.Image.open(img3d_file), dtype=np.uint8).reshape((720, 1280, 3))
 
	# # 3d 박스 이미지 좌표계에 투영
	# img2 = box3d_projection(copy.deepcopy(img))
	# # 3d 박스안에 있는 포인트만 이미지 좌표계에 투영
	# img3 = pnts_in_box_projection(copy.deepcopy(img2), dist_thres=0.1, score_thres=0.5)
	# # 2d 박스 이미지에 플롯
	# img4 = box2d_visualize(copy.deepcopy(img2))
 
	fig = plt.figure(figsize=(20,20))
	# ax1 = fig.add_subplot(2, 2, 1)
	# ax1.imshow(img)
	# ax1.set_title('raw image')
	# ax1.axis("off")
	"""
	If it's needed to make a subplot of a custom size, one way is to pass (left, bottom, width, height) information to a add_axes() call on the figure object.
	"""
	# ax2 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_axes([0.05, 0.31, 0.9, 0.8])
	ax2.imshow(img3d)
	ax2.set_title('rgbd points+ lidar points + 3D boxes')
	ax2.axis("off")

	# ax3 = fig.add_subplot(2, 2, 3)
	ax3 = fig.add_axes([0.01, 0, 0.45, 0.3])
	ax3.imshow(img)
	ax3.set_title('color image')
	# ax3.imshow(img3)
	# ax3.set_title('projected points and boxes')
	ax3.axis("off")

	# ax4 = fig.add_subplot(2, 2, 4)
	ax4 = fig.add_axes([0.54, 0, 0.45, 0.3])
	ax4.imshow(depth_raw)
	ax4.set_title('depth image')
	# ax4.imshow(img4)
	# ax4.set_title('2D boxes and projected boxes')
	ax4.axis("off")
 
	fig.savefig(args.save_to + "/" + str(input_output["file_name"][i].with_suffix('.png')))
	plt.cla()
	plt.clf()
	plt.close(fig)

