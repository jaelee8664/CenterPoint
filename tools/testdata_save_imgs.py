import argparse
import json
import os
import sys
import time
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
from pathlib import Path
import yaml
from det3d import torchie
import matplotlib.pyplot as plt
import cv2
import PIL
import open3d as o3d
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint

from yoloX.yolox_utils import get_model, fuse_model, preproc, postprocess, visual

import open3d as o3d
from open3d.web_visualizer import draw
import copy

# 주피터 노트북 경로설정
os.chdir('../')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", default="configs/onlypoints/pp/onlypoints_centerpoint_pp.py", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
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
    parser.add_argument(
        "--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
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
logger = get_root_logger(cfg.log_level)
logger.info("Distributed training: {}".format(distributed))
logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
args.testset = False
dataset = build_dataset(cfg.data.test)
args.speed_test = True
data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
args.checkpoint = "modelzoo/etri3D_pointpillar/etri3D_latest.pth"
checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
model = model.cuda()
model.eval()
mode = "test"
# YOLOX
# yolox_pth_path = "modelzoo/yolox/yolox_tiny_idgranter.pth.tar"
yolox_pth_path = "modelzoo/yolox/ocsort_x_mot20.pth.tar"
device = "cuda:0"
ckpt = torch.load(yolox_pth_path, map_location=device)

model2D = get_model(depth=1.33, width=1.25).to(device)
model2D.load_state_dict(ckpt["model"])
model2D = fuse_model(model2D)

class Predictor():
    """
    YOLOX inference calss
    """
    def __init__(self, model, image_size, device, nms_thre):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.nms_thre = nms_thre
        self.campos = 0

    def inference(self, img):
        """
        Inference bounding boxes
        Args:
            img (array): images

        Returns:
            bounding boxes
        """
        x, _ = preproc(img, self.image_size, self.mean, self.std)
        x = torch.from_numpy(x).unsqueeze(0).float().to(self.device)
        output = self.model(x)
        output2 = postprocess(output, num_classes=1,
                              conf_thre=0.1, nms_thre=self.nms_thre)

        return output2
    
nms_thre = 0.5
test_size = (800, 1440)
predictor = Predictor(model2D, test_size, device, nms_thre)
# img = np.ascontiguousarray(img, dtype=np.uint8)
input_output = {"file_name": [], "points": [], "gt_boxes": [], "gt_classes": [], "output_boxes": [], "output_classes": [], "output_scores": [], "2D_states": []}
for i, data_batch in tqdm(enumerate(data_loader)):
	if i == start:
		torch.cuda.synchronize()
		time_start = time.time()

	if i == end:
		torch.cuda.synchronize()
		time_end = time.time()

	with torch.no_grad():
		outputs = batch_processor(
			model, data_batch, train_mode=False, local_rank=args.local_rank,
		)
	for j, output in enumerate(outputs):
		input_output["points"].append(data_batch['points'][j].cpu().numpy())
		input_output["file_name"].append(data_batch['metadata'][j]['filename'])
		file = Path("data/test/camera")/data_batch['metadata'][j]['filename'].with_suffix('.png')
		img = np.array(PIL.Image.open(file), dtype=np.uint8)
		with torch.no_grad():
			outputs_2D = predictor.inference(img[..., :3])
		input_output["2D_states"].append(outputs_2D[0])
		# gt_boxes = []
		# for ts in data_batch['anno_box']:
		#     ts2 = ts[j]
		#     gt_boxes.extend(ts2.cpu().numpy())
		# input_output["gt_boxes"].append(np.array(gt_boxes))
		# gt_classes = []
		# idx = 0
		# for k in range(data_batch["anno_cls"][j].shape[0]):
		#     head_classes = cfg.tasks[k]
		#     class_names = head_classes['class_names']
		#     head_class_ids = data_batch['anno_cls'][j][k]
		#     for l in range(len(head_class_ids)):
		#         _id = head_class_ids[l] - 1
		#         gt_classes.append(idx+_id)
		#     idx += len(class_names)
		input_output["output_classes"].append(output['label_preds'].cpu().numpy())
		# input_output["gt_classes"].append(np.array(gt_classes))
		input_output["output_boxes"].append(output["box3d_lidar"].cpu().numpy())
		input_output["output_scores"].append(output['scores'].cpu().numpy())

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

calib = np.array([[-0.99983949, -0.01008867, -0.01480574, 0.05565861],
[0.01433594, 0.04514753, -0.99887746, 0.08249057],
# [0.01074579, -0.99892939, -0.04499565, -0.00898347],
[0.01074579, -0.99892939, -0.04499565, 0.8-0.00898347],
[0, 0, 0, 1]])
offset = [[1,0,0,0], [0,1,0,0], [0,0,1,0.8],[0,0,0,1]]
# calib = np.linalg.inv(calib)
cam = np.array([[1076.7392578125, 0.0, 957.8059692382812],
[0.0, 1076.7392578125, 546.310791015625],
[0.0, 0.0, 1.0]])
# cam = [[1072.7210693359375, 0.0, 980.5731201171875],
# [0.0, 1072.7210693359375, 554.7196044921875],
# [0.0, 0.0, 1.0]]
for i in tqdm(range(len(input_output["points"]))):
	points_v = input_output["points"][i]
	pred_boxes = input_output["output_boxes"][i]
	scores = input_output["output_scores"][i]
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points_v)
	entities_to_draw = [pcd, mesh_frame]
	boxes3d_pts_list = []
 
	# pred_boxes 그리기
	for idx in range(len(pred_boxes)):
		# if scores[idx] < 0.3:
		# 	continue
		if input_output["output_classes"][i][idx] != 8:
			continue
		translation = pred_boxes[idx][:3]
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
		box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
		box.color = [1, 0, 0]           #Box color would be red box.color = [R,G,B]
		entities_to_draw.append(box)

	file = Path("data/test/camera")/input_output["file_name"][i].with_suffix('.png')
	img = np.array(PIL.Image.open(file), dtype=np.uint8)
	fig = plt.figure(figsize=(20,20))
	img2 = copy.deepcopy(img)
	img3 = copy.deepcopy(img)
	img4 = copy.deepcopy(img)
 
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
			cv2.line(img2, (boxes3d_pts_list[j][pnts[0], 0].astype(np.int32), boxes3d_pts_list[j][pnts[0], 1].astype(np.int32)), (boxes3d_pts_list[j][pnts[1], 0].astype(np.int32), boxes3d_pts_list[j][pnts[1], 1].astype(np.int32)), (255, 0, 0), thickness = 5)
			cv2.line(img4, (boxes3d_pts_list[j][pnts[0], 0].astype(np.int32), boxes3d_pts_list[j][pnts[0], 1].astype(np.int32)), (boxes3d_pts_list[j][pnts[1], 0].astype(np.int32), boxes3d_pts_list[j][pnts[1], 1].astype(np.int32)), (255, 0, 0), thickness = 5)
	ax1 = fig.add_subplot(2, 2, 1)
	ax1.imshow(img)
	ax1.set_title('raw image')
	ax1.axis("off")
	
	ax2 = fig.add_subplot(2, 2, 2)
	ax2.imshow(img2)
	ax2.set_title('projected 3D boxes')
	ax2.axis("off")
 
	points_v_appended =  np.concatenate((points_v, np.ones((points_v.shape[0], 1))), axis=1)
	cam_points = np.matmul(cam, np.matmul(np.matmul(calib, offset), points_v_appended.T).T[:, :3].T).T
	for k in range(cam_points.shape[0]):
		cam_points[k, 0] = cam_points[k, 0] / cam_points[k, 2]
		cam_points[k, 1] = cam_points[k, 1] / cam_points[k, 2]
		cam_points[k, 2] = 1
	for l in range(cam_points.shape[0]):
		pnts = cam_points[l, :]
		if 0 <= pnts[0].astype(np.int32) < 1920 and 0 <= pnts[1].astype(np.int32) < 1080:
			cv2.line(img3, (pnts[0].astype(np.int32), pnts[1].astype(np.int32)), (pnts[0].astype(np.int32), pnts[1].astype(np.int32)), (255, 0, 0), thickness = 5)
	ax3 = fig.add_subplot(2, 2, 3)
	ax3.imshow(img3)
	ax3.set_title('projected points')
	ax3.axis("off")
	if input_output["2D_states"][i] is not None:
		result_img_rgb, bboxes, scores, cls, cls_names = visual(input_output["2D_states"][i], img4, test_size, cls_conf=0.5)
	else:
		result_img_rgb = img4

	ax4 = fig.add_subplot(2, 2, 4)
	ax4.imshow(img4)
	ax4.set_title('2D boxes and projected 3D boxes')
	ax4.axis("off")
	fig.savefig("/home/jaelee/objdect/CenterPoint/plot_results/" + str(input_output["file_name"][i].with_suffix('.png')))