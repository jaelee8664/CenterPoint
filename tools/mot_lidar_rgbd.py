import argparse
import os
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
import PIL
from tqdm import tqdm

# 2D 모델 함수
from yoloX.yolox_utils import get_model, fuse_model, preproc, postprocess2, visual
# 3D 모델 함수
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.pipelines import Compose, PreprocessRealtime
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import batch_processor
from det3d.torchie.trainer import load_checkpoint

# MOT(Multi Object Tracking)
from det3d.utils.tracker_utils import l2norm_batch, Track

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    # parser.add_argument("--config", default="configs/onlypoints/pp/onlypoints_centerpoint_pp.py", help="train config file path")
    parser.add_argument("--config", default="configs/onlypoints/pp/realtime_centerpoint_pp.py", help="train config file path")
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
    parser.add_argument("--pth_dir2d", type=str, default="modelzoo/yolox/ocsort_x_mot20.pth.tar")
    parser.add_argument("--pth_dir3d", type=str, default="modelzoo/etri3D_pointpillar/etri3D_latest.pth")
    args = parser.parse_args(args=[])
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args

# def only_person(output2d):
#     """
#     Args:
#         output2d (np.array): 2d 박스 x1y1x2y2 상태와 score, class 정보를 담은 N x 7 array
#     Return:
#         output2d (np.array): 사람(person), 자전거(bicycle) 클래스 정보만을 포함한 2d bounding boxes
#     """
#     mask_person = output2d[..., 6] == 0
#     mask_bicycle = output2d[..., 6] == 1
#     mask = torch.logical_and(mask_bicycle, mask_person)
#     output2d = output2d[mask]
#     return output2d

class DataPusher():
    def __init__(self, root_path):
        self.pcd_file_name = []
        self.root_path = Path(root_path)
        for (root, dirs, file) in os.walk(str(self.root_path)):
                if root == str(self.root_path) + "/pcd":
                    for f in file:
                        self.pcd_file_name.append(Path(f))
        self.pcd_file_name.sort()
    
    def __len__(self):
        return len(self.pcd_file_name)
    
    def pusher(self, i):
        """
        Args:
            i (int): 인덱스

        Returns:
            rgb_image: rgb 이미지, H x W x 3 numpy array 
            depth_image: depth 이미지, H x W numpy array
            points: point cloud, N x 3 numpy array
        """
        pcd_file = self.root_path / Path("pcd") / Path(self.pcd_file_name[i])
        rgbd_dir = self.root_path / Path("rgbd") / Path(self.pcd_file_name[i])
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        points = np.asarray(pcd.points, dtype=np.float32)
        rgb_file = str(self.root_path / Path("rgbd") /rgbd_dir.stem) + "_rgb.jpg"
        rgb_image = np.array(PIL.Image.open(rgb_file), dtype=np.uint8)
        depth_file = str(self.root_path / Path("rgbd") /rgbd_dir.stem) + "_d.png"
        depth_image = np.array(PIL.Image.open(depth_file), dtype=np.uint16) / 200
        return rgb_image, depth_image, points
        

class Detector():
    # 2D + 3D 디텍터 클래스
    def __init__(self, args, nms_thres, score_thres2d, score_thres3d, num_classes = 1):
        # 공통 정보
        self.device = "cuda:0"
        cfg = Config.fromfile(args.config)
        
        # 2D object detection(yolox) 모델 클래스 변수
        self.image_size = (800, 1440) # 학습한 사진 크기
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.nms_thres = nms_thres # 2d box NMS threshold
        self.score_thres2d = score_thres2d # 2d box score threshold
        self.pth_dir2d = args.pth_dir2d # "modelzoo/yolox/ocsort_x_mot20.pth.tar"
        
        self.model2d = get_model(depth=1.33, width=1.25, num_classes = num_classes).to(self.device) # yolox 모델
        ckpt = torch.load(args.pth_dir2d, map_location=self.device)
        self.model2d.load_state_dict(ckpt["model"])
        self.model2d = fuse_model(self.model2d)
        self.num_classes = num_classes
        # 3D object detection(centerpoint) 모델 클래스 변수
        self.model3d = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg) # centerpoint 모델
        self.score_thres3d = score_thres3d # 3d box score threshold
        self.pth_dir3d = args.pth_dir3d # "modelzoo/etri3D_pointpillar/etri3D_latest.pth"
        self.pipeline = PreprocessRealtime(device=self.device, cfg=cfg.voxel_generator)
        self.local_rank = args.local_rank
        self.model3d = self.model3d.to(self.device)
        # print(next(self.model3d.parameters()).device)
        load_checkpoint(self.model3d, self.pth_dir3d, map_location=self.device)
    def inference(self, image, points):
        """
        Args:
            image (np.array): rgb 이미지, H x W x 3 numpy array 
            points (np.array): point cloud, N x 3 numpy array
        Returns:
            output2d[0] (np.array): 2d 박스 x1y1x2y2 상태와 score, class 정보를 담은 N x 6 array
            output3d[0] (dict): {'label_preds': torch.array, 'box3d_lidar': torch.array, 'scores' : torch.array}
                'label_preds': 3d 박스 클래스, N array
                'box3d_lidar': 3d 박스 형상 [x, y, z, w, l, h, rot] N x 7 array
                'scores': 3d 박스 cofidence score, N array
        """
        # 2D object detection Inference
        output2d, _ = preproc(image, self.image_size, self.mean, self.std)
        output2d = torch.from_numpy(output2d).unsqueeze(0).float().to(self.device)
        output2d = self.model2d(output2d)
        # output2d = postprocess(output2d, num_classes=self.num_classes, conf_thre=self.score_thres2d, nms_thre=self.nms_thres)
        output2d = postprocess2(output2d, self.num_classes, self.image_size, image.shape, conf_thre=self.score_thres2d, nms_thre=self.nms_thres)
        
        # 3D object detection Inference
        input3d = self.pipeline(points)
        
        with torch.no_grad():
            output3d = batch_processor(
			self.model3d, input3d, train_mode=False, local_rank=self.local_rank)
        
        return output2d[0], output3d[0] # batch size = 1 이므로 0으로 인덱싱

class Tracker():
    # Multi Object Tracking 모듈
    def __init__(self):
        self.tracks = [] # 트랙 보관 리스트
        self.min_hits = 3 # 트랙 초기화를 위한 최소 검출 횟수
        self.match3d_thres = 0.5 # 매칭을 위한 최대 매칭 기준 (m)
        self.min_hits = 3 # 트랙 초기화시 매칭되어야하는 최소 횟수
        self.t = 0.25 # 타임스텝
        self.ret = [] # 현재 신뢰하는 트랙 상태
        self.frame_count = 0 # 총 지난 횟수
        self.max_age = 30 # 트랙이 제거되는 언매칭 횟수 기준. max_age동안 매칭이 되지 않으면 self.tracks에서 트랙 제거
        
        # 카메라 내부, 외부 파라미터
        
    def matching(self, detections2d, detections3d, depth_image):
        """
        Args:
            detections2d (torch.array): 2d 박스 x1y1x2y2 상태와 score, class 정보를 담은 N x 6 array
            detections3d (dict of torch.array): {'label_preds': torch.array, 'box3d_lidar': torch.array, 'scores' : torch.array}
                'label_preds': 3d 박스 클래스, N array
                'box3d_lidar': 3d 박스 형상 [x, y, z, w, l, h, rot] N x 6 array
                'scores': 3d 박스 cofidence score, N array
            depth_image: (np.array) W x H 뎁스 이미지
        Returns:
            self.ret: 현재 신뢰하는 트랙정보
        """
        self.ret.clear()
        # 1. 2D measurement + 3D measurement 매칭
        
        # 2. 3D measurement + track 매칭
        # 3. 2D measurement + track 매칭
        # 4. 남은 2D measurement 초기화
        unmatched_dets = []
        trk_idx = len(self.tracks)
        for i in unmatched_dets:
            trk = Track(detections2d[i, :], self.t)
            self.tracks.append(trk)
        for trk in reversed(self.tracks):
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count < self.min_hits):
               self.ret.append(np.concatenate((np.array([[trk.id]]), trk.get_state()), axis=1))
            trk_idx -= 1
            if trk.time_since_update > self.max_age:
                self.tracks.pop(trk_idx)
        self.frame_count += 1
        return self.ret
    
    def get_measurements_from2d():
        
        pass
    
    def get_tracks(self):
        """
        현재 존재하는 track 반환
        """
        return self.ret

def main():
    args = parse_args()
    dataset = DataPusher(root_path="data/kitech")
    detector = Detector(args, nms_thres = 0.5, score_thres2d = 0.7, score_thres3d = 0.1)
    tracker = Tracker()
    
    for i in tqdm(range(len(dataset))):
        # 1186 사람 첫 등장
        rgb_image, depth_image, pcd = dataset.pusher(i)
        output2d, output3d = detector.inference(rgb_image, pcd)
        # # 사람, 자전거만 detecting
        if output2d is not None:
            print("2D detected")
        cur_tracks = tracker.matching(output2d, output3d, depth_image)
        
        
if __name__ == "__main__":
    main()