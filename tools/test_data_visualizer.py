# 실행코드예시: python3 tools/test_data_visualizer.py --work_dir work_dirs/test --load_from tools/vis_utils/pred_data.pickle
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", default="configs/onlypoints/pp/onlypoints_centerpoint_pp.py", help="train config file path")
    parser.add_argument("--work_dir", type=str, help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--load_from", type=str, help="load saved pickle file")
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

os.makedirs(args.work_dir, exist_ok=True)

if os.path.exists(args.load_from):
    with open(args.load_from, "rb") as f:
        input_output = pickle.load(f)
else:
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

    args.checkpoint = "modelzoo/etri3D_latest.pth"
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
    with open(args.load_from, "wb") as f:
        pickle.dump(input_output, f)

# visualizer
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
pcd = o3d.geometry.PointCloud()

# Open3D Visualizer
vis = o3d.visualization.Visualizer()
# vis.create_window()
vis.create_window(width=1280, height=720)  # Set the window size as per your requirement

vis.get_render_option().point_size = 1.0  # Set point size

# Set camera parameters
ctr = vis.get_view_control()
parameters = o3d.io.read_pinhole_camera_parameters("tools/vis_utils/ScreenCamera_2024-04-17-13-22-54.json")

cnt = 0
# Set a uniform color for all points
uniform_color = [0, 0, 1]  # Blue color, you can change it to any other color you like

for idx1 in range(len(input_output["file_name"])):

    points_v = input_output["points"][idx1]
    pred_boxes = input_output["output_boxes"][idx1]
    scores = input_output["output_scores"][idx1]
    boxes3d_pts_list = []

    file = Path("data/test/camera")/input_output["file_name"][idx1].with_suffix('.png')
    img = np.array(PIL.Image.open(file), dtype=np.uint8)
    pcd.points = o3d.utility.Vector3dVector(points_v)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(uniform_color, (len(points_v), 1)))  # Set uniform color for all points
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    # entities_to_draw = [pcd, mesh_frame]
    for idx2 in range(len(pred_boxes)):
        # if scores[idx] < 0.1:
        #         continue
        if input_output["output_classes"][idx1][idx2] != 8:
                continue
        translation = pred_boxes[idx2][:3]
        w, l, h = pred_boxes[idx2][3], pred_boxes[idx2][4], pred_boxes[idx2][5]
        rotation = pred_boxes[idx2][-1]

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
        boxes3d_pts = boxes3d_pts.T
        boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts.T)
        boxes3d_pts_list.append(np.asarray(boxes3d_pts))
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

