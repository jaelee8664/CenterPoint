# 실행코드예시: python3 tools/test_data_visualizer.py --work_dir work_dirs/offset_minus0p8_50min --load_from tools/vis_utils/offset_minus0p8_50min.pickle --checkpoint modelzoo/etri3D_latest_50min.pth --z_offset -0.8
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
import PIL
import numpy as np
import torch
from pathlib import Path
import yaml
from det3d import torchie
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
import torch.distributed as dist
import subprocess
import pickle
import open3d as o3d

cam = np.zeros([3,3])
cam[0,0] = 262.49102783203125
cam[1,1] = 262.49102783203125
cam[2,2] = 1
cam[0,2] = 327.126708984375
cam[1,2] = 184.74203491210938
calib = np.array([[ 0, -1,  0,  0.02],
                  [ 0,  0, -1, -0.17],
                  [ 1,  0,  0, -0.06]])
calib4 = np.concatenate((calib, np.array([[0,0,0,1]])), axis=0)
inv_camlib4 = np.linalg.inv(calib4)
inv_cam = np.linalg.inv(cam)
offset = np.identity(4)
test_size = (800, 1440)
def z_offset(data_batch, val):
    for points in data_batch["points"]:
        points[:,2] += val
    data_batch["voxels"][..., 2] += val

def pnts_to_lidar(color_img, depth_img, inv_cam, inv_camlib4):
    depth_states = np.ones((4, depth_img.shape[0] * depth_img.shape[1] ))
    idx = 0
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            k = depth_img[i, j]
            depth_states[0, idx] = j
            depth_states[1, idx] = i
            depth_states[:3, idx] = depth_states[:3, idx] * k
            idx += 1
    depth_states[:3, :] = np.matmul(inv_cam, depth_states[:3, :])
    depth_states = np.matmul(inv_camlib4, depth_states)
    color_states = color_img.reshape(color_img.shape[2], color_img.shape[0]*color_img.shape[1]).astype(np.float32) / 255
    return color_states, depth_states

def img_to_pointcloud(img, depth, K, Rt):
    rgb = o3d.geometry.Image(img)
    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=200.0, depth_trunc=1000, convert_rgb_to_intensity=False)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(img.shape[1], img.shape[0], fx, fy, cx, cy)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(2*fx), int(2*fy), fx, fy, cx, cy)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, Rt)
    return pc


def pnt2d_img_to_lidar(img_states, depth_img, inv_cam, inv_camlib4, test_size, score_thres = 0.5):
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
    scores = img_states[:, 4] * img_states[:, 5]
    x_states = (bboxes[:, 0] + bboxes[:, 2]) // 2
    y_states = (bboxes[:, 1] + bboxes[:, 3]) // 2
    lidar_states[0, :] = x_states.T
    lidar_states[1, :] = y_states.T
    for i in range(img_states.shape[0]):
        if 0 <= x_states[i] < depth_img.shape[1] and 0 <= y_states[i] < depth_img.shape[0] and scores[i] > score_thres:
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
    parser.add_argument("--work_dir", default = "work_dirs/kitech_data", type=str, help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--load_from", default = "kitech_inputoutput_result.pickle", type=str, help="load saved pickle file")
    parser.add_argument("--z_offset", type=float, default=0.0, help="z offset")
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
    parser.add_argument("--checkpoint",default="modelzoo/etri3D_pointpillar/etri3D_latest.pth", type=str, help="pretrained model path")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args
args = parse_args()
cfg = Config.fromfile(args.config)
os.makedirs(args.work_dir, exist_ok=True)

# if os.path.exists(args.load_from):
#     with open(args.load_from, "rb") as f:
#         input_output = pickle.load(f)
# else:
#     raise NotImplementedError

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

checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

model = model.cuda()
model.eval()
mode = "test"

detections = {}
cpu_device = torch.device("cpu")

start = time.time()

start = int(len(dataset) / 3)
end = int(len(dataset) * 2 /3)

time_start = 0 
time_end = 0 

input_output = {"file_name": [], "points": [], "gt_boxes": [], "gt_classes": [], "output_boxes": [], "output_classes": [], "output_scores": []}
for i, data_batch in tqdm(enumerate(data_loader)):
    if args.z_offset != 0:
        z_offset(data_batch, args.z_offset)
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


# visualizer
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
pcd = o3d.geometry.PointCloud()
# pcd2 = o3d.geometry.PointCloud()
# Open3D Visualizer
vis = o3d.visualization.Visualizer()

# vis.create_window()
vis.create_window(width=1280, height=720)  # Set the window size as per your requirement
vis.get_render_option().point_size = 1.0  # Set point size
vis.get_render_option().line_width = 10

# Set camera parameters
ctr = vis.get_view_control()
parameters = o3d.io.read_pinhole_camera_parameters("tools/vis_utils/ScreenCamera_2024-04-30-09-53-54.json")

cnt = 0
# Set a uniform color for all points
uniform_color = [0, 0, 1]  # Blue color, you can change it to any other color you like

for idx1 in range(len(input_output["file_name"])):
    # if idx1 < 3450:
    #     continue
    points_v = input_output["points"][idx1]
    pred_boxes = input_output["output_boxes"][idx1]
    scores = input_output["output_scores"][idx1]
    boxes3d_pts_list = []

    # file = Path(cfg.data_root)/Path("camera")/input_output["file_name"][idx1].with_suffix('.png')
    rgb_name = Path(str(input_output["file_name"][idx1].stem) + "_rgb.jpg")
    depth_name = Path(str(input_output["file_name"][idx1].stem) + "_d.png")
    rgb_file = Path(cfg.data_root)/Path("rgbd")/rgb_name
    depth_file = Path(cfg.data_root)/Path("rgbd")/depth_name
    img = np.array(PIL.Image.open(rgb_file), dtype=np.uint8)
    depth_raw = np.array(PIL.Image.open(depth_file), dtype=np.uint16)
    depth = depth_raw / 200
    pcd.points = o3d.utility.Vector3dVector(points_v)
    pcd_col = np.tile(uniform_color, (len(points_v), 1))
    pcd.colors = o3d.utility.Vector3dVector(pcd_col)  # Set uniform color for all points
    
    # dpnt_colors, dpnts = pnts_to_lidar(img, depth, inv_cam, inv_camlib4)
    # pcd2.points = o3d.utility.Vector3dVector(dpnts[:3, :].T)
    # pcd2.colors = o3d.utility.Vector3dVector(dpnt_colors.T)
    pcd2 = img_to_pointcloud(img, depth_raw, cam, calib4)
    
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.add_geometry(pcd2)
    vis.add_geometry(mesh_frame)


    # for idx2 in range(len(pred_boxes)):
    #     if scores[idx2] < 0.9:
    #             continue
    #     if input_output["output_classes"][idx1][idx2] != 8:
    #             continue
    #     translation = pred_boxes[idx2][:3]
    #     w, l, h = pred_boxes[idx2][3], pred_boxes[idx2][4], pred_boxes[idx2][5]
    #     rotation = pred_boxes[idx2][-1]

    #     bounding_box = np.array([
    #                     [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
    #                     [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
    #                     [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]]) 
    #     rotation_matrix = np.array([
    #             [np.cos(rotation), -np.sin(rotation), 0.0],
    #             [np.sin(rotation), np.cos(rotation), 0.0],
    #             [0.0, 0.0, 1.0]])
    #     eight_points = np.tile(translation, (8, 1))

    #     corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    #     boxes3d_pts = corner_box.transpose()
    #     boxes3d_pts = boxes3d_pts.T
    #     boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts.T)
    #     boxes3d_pts_list.append(np.asarray(boxes3d_pts))
    #     box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
    #     box.color = [0, 1, 0]           #Box color would be red box.color = [R,G,B]
    #     vis.add_geometry(box)

    # # 2D_boxes 그리기
    if input_output["2D_states"][idx1] is not None:
        # print(input_output["2D_states"][idx1])
        depth_boxes = pnt2d_img_to_lidar(input_output["2D_states"][idx1], depth, inv_cam, inv_camlib4, test_size)[:3, :].T
        # pred_boxes 그리기
        print(depth_boxes)
        for idx2 in range(depth_boxes.shape[0]):
            if depth_boxes[idx2][0] == 0.06: # 라이다 원점을 중심으로 하는 박스는 제거
                continue
            w = l = h = 1
            bounding_box = np.array([
                            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
            translation = depth_boxes[idx2, :]
            eight_points = np.tile(translation, (8, 1))
            rotation = np.array([[ 0.59587538,  0.80307692,  0.   ],
                            [-0.80307692,  0.59587538,  0.        ],
                            [ 0.,          0.,          1.        ]])
            corner_box = np.dot(rotation, bounding_box) + eight_points.transpose()
            boxes3d_pts = corner_box.transpose()
            # boxes3d_pts[:, 2] = np.clip(boxes3d_pts[:, 2], -1.6, 60) # 박스의 아랫부분을 잘라내기 위한 클리핑
            boxes3d_pts = boxes3d_pts.T
            
            boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts.T)
            # boxes3d_pts_list.append(np.asarray(boxes3d_pts))
            box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
            box.color = [1, 0, 0]           #Box color would be red box.color = [R,G,B]
            vis.add_geometry(box)    

    vis.poll_events()
    ctr.convert_from_pinhole_camera_parameters(parameters, True)
    vis.update_renderer()
    vis.run()
    a = args.work_dir + "/" + str(input_output["file_name"][idx1].with_suffix('.png'))
    print(a)
    vis.capture_screen_image(a)
vis.destroy_window()

