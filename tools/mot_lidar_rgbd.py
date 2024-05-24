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
from det3d.utils.tracker_utils import l2norm_batch, linear_assignment, Track

# global variable
MAXVAL = 1000000000

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
        self.match3d_thres = 1. # 매칭을 위한 최대 매칭 기준 (m)
        self.min_hits = 3 # 트랙 초기화시 매칭되어야하는 최소 횟수
        self.t = 0.25 # 타임스텝
        self.ret = [] # 현재 신뢰하는 트랙 상태
        self.frame_count = 0 # 총 지난 횟수
        self.max_age = 30 # 트랙이 제거되는 언매칭 횟수 기준. max_age동안 매칭이 되지 않으면 self.tracks에서 트랙 제거
        
        # 카메라 내부, 외부 파라미터
        self.intrinsic = np.array([[262.49102783203125, 0, 327.126708984375],
                             [0, 262.49102783203125, 184.74203491210938],
                             [0, 0, 1]])
        
        self.extrinsic = np.array([[ 0, -1,  0,  0.02],
                    [ 0,  0, -1, -0.17],
                    [ 1,  0,  0, -0.06]])
        self.extrinsic4 = np.concatenate((self.extrinsic, np.array([[0,0,0,1]])), axis=0)
        self.inv_intrinsic = np.linalg.inv(self.intrinsic)
        self.inv_extrinsic4 = np.linalg.inv(self.extrinsic4)
        
    def pnt2d_img_to_lidar(self, img_states, depth_img):
        """
        Args:
            img_states (np.array): 2d 박스 x1y1x2y2 상태와 score, class 정보를 담은 N x 6 array
        Returns:
            2d box 가운데의 lidar 좌표계에서의 위치, 클래스 [x, y, z] N x 3 array
        """
        lidar_states = np.ones((4, img_states.shape[0]))
        # lidar_states[3, :] = img_states[:, 5].T
        # img_states = img_states.cpu().numpy()
        # bboxes = img_states[:, 0:4]

        x_states = (img_states[:, 0] + img_states[:, 2]) // 2
        y_states = (img_states[:, 1] + img_states[:, 3]) // 2
        lidar_states[0, :] = x_states.T
        lidar_states[1, :] = y_states.T
        for i in range(img_states.shape[0]):
            if 0 <= x_states[i] < depth_img.shape[1] and 0 <= y_states[i] < depth_img.shape[0]:
                z = depth_img[int(y_states[i]), int(x_states[i])]
                lidar_states[:3, i] *= z
            else:
                lidar_states[:, i] = MAXVAL # depth를 뽑지 못하는 위치는 MAXVAL로 기록
        lidar_states[:3, :] = np.matmul(self.inv_intrinsic, lidar_states[:3, :])
        lidar_states = np.matmul(self.inv_extrinsic4, lidar_states)
        
        return lidar_states.T[:, :3]

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
        
        # torch to numpy
        bbox3d = detections3d['box3d_lidar'].cpu().numpy()
        score3d = detections3d['scores'].cpu().numpy()
        class3d = detections3d['label_preds'].cpu().numpy()
        combined_detections = np.empty((0, 3)) # 2D + 3D 합쳐진 measurement
        bbox2d_glob = np.empty((0, 3))
        if detections2d is not None: 
            detections2d = detections2d.detach().numpy()
            bbox2d = detections2d[:, :4]
            # 1. 2D measurement + 3D measurement 매칭
            bbox2d_glob = self.pnt2d_img_to_lidar(bbox2d, depth_image) # 2D 위치 => 3D 위치
            measure_matrix = l2norm_batch(bbox2d_glob, bbox3d)
            matched_idx = linear_assignment(measure_matrix)
            
            
            to_remove_detections2d = []
            to_remove_detections3d = []
            for m in matched_idx:
                idx2d, idx3d = m[0], m[1]
                if measure_matrix[m[0], m[1]] >= self.match3d_thres:
                    continue
                to_remove_detections2d.append(idx2d)
                to_remove_detections3d.append(idx3d)
            combined_detections = bbox3d[np.array(to_remove_detections3d, dtype=np.int64)] # 3D 위치정보가 더 정확하다 가정
            remained_idx_2d = np.setdiff1d(np.arange(bbox2d_glob.shape[0]), np.array(to_remove_detections2d))
            remained_idx_3d = np.setdiff1d(np.arange(bbox3d.shape[0]), np.array(to_remove_detections2d))
            bbox2d_glob = bbox2d_glob[remained_idx_2d]
            bbox3d = bbox3d[remained_idx_3d]
            score3d = score3d[remained_idx_3d]
            class3d = class3d[remained_idx_3d]
        
        # prediction step
        trks = np.zeros((len(self.tracks), 3))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict(self.t)
            trk[:] = [pos[0], pos[1], pos[2]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)
        
        # 2. 3D measurement + track 매칭
        unmatched_tracks = []
        unmatched_combined_detections = []
        matched_tracks = []
        # 2-1. combined measurement 매칭
        if combined_detections.shape[0] > 0:
            measure_matrix = l2norm_batch(combined_detections, trks)
            matched_idx = linear_assignment(measure_matrix)
            # for d, _ in enumerate(combined_detections):
            #     if (d not in matched_idx[:, 0]):
            #         unmatched_combined_detections.append(d)
            # for t, _ in enumerate(trks):
            #     if (t not in matched_idx[:, 1]):
            #         unmatched_tracks.append(t)
            for m in matched_idx:
                if measure_matrix[m[0], m[1]] >= self.match3d_thres:
                    unmatched_combined_detections.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matched_tracks.append(m.reshape(1,2))
            for m in matched_tracks:
                self.tracks[m[1]].update(combined_detections[m[0], :])
        else:
            for t, _ in enumerate(trks):
                unmatched_tracks.append(t)
                matched_tracks = np.empty((0, 2), dtype=int)

        unmatched_tracks = np.array(unmatched_tracks)
        unmatched_combined_detections = np.array(unmatched_combined_detections)
        # 2-2. 3D measurement 매칭
        if bbox3d.shape[0] > 0 and unmatched_tracks.shape[0] > 0:
            left_tracks = trks[unmatched_tracks]
            measure_matrix = l2norm_batch(bbox3d, left_tracks)
            matched_idx = linear_assignment(measure_matrix)
            to_remove_tracks = []
            for m in matched_idx:
                det_idx, trk_idx = m[0], unmatched_tracks[m[1]]
                if measure_matrix[m[0], m[1]] >= self.match3d_thres:
                    continue
                self.tracks[m[1]].update(bbox3d[m[0], :])
                to_remove_tracks.append(trk_idx)
            unmatched_tracks = np.setdiff1d(unmatched_tracks, np.array(to_remove_tracks))
        
        # 3. 2D measurement + track 매칭
        unmatched_detection2d = []
        if bbox2d_glob.shape[0] > 0 and unmatched_tracks.shape[0] > 0:
            left_tracks = trks[unmatched_tracks]
            measure_matrix = l2norm_batch(bbox2d_glob, left_tracks)
            matched_idx = linear_assignment(measure_matrix)
            to_remove_tracks = []
            to_remove_detections2d = []
            for m in matched_idx:
                det_idx, trk_idx = m[0], unmatched_tracks[m[1]]
                if measure_matrix[m[0], m[1]] >= self.match3d_thres:
                    continue
                self.tracks[m[1]].update(bbox2d_glob[m[0], :])
                to_remove_tracks.append(trk_idx)
                to_remove_detections2d.append(det_idx)
            unmatched_tracks = np.setdiff1d(unmatched_tracks, np.array(to_remove_tracks))
            for d, _ in enumerate(bbox2d_glob):
                if (d not in matched_idx[:, 0]):
                    unmatched_detection2d.append(d)
        else:
            unmatched_detection2d = np.array([i for i in range(bbox2d_glob.shape[0])])
            
        # 4. 남은 2D measurement 초기화
        trk_idx = len(self.tracks)
        for i in unmatched_detection2d:
            trk = Track(bbox2d_glob[i, :], self.t)
            self.tracks.append(trk)
        for i in unmatched_combined_detections:
            trk = Track(combined_detections[i, :3], self.t)
            self.tracks.append(trk)
        for trk in reversed(self.tracks):
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count < self.min_hits):
                self.ret.append(np.concatenate((np.array([trk.id]), trk.get_state()), axis=0))
            trk_idx -= 1
            if trk.time_since_update > self.max_age:
                self.tracks.pop(trk_idx)
        self.frame_count += 1
        return self.ret
    
    def get_his_tracks(self):
        """
        현재 신뢰하는 track의 과거경로 반환
        """
        cur_id_list = list(self.ret[0, :])
        his = []
        for trk in self.tracks:
            if trk.id in cur_id_list:
                his.append([trk.id, trk.get_history()])
        return his
    
    def get_cur_tracks(self):
        """
        현재 신뢰하는 track 반환
        """
        return self.ret

def main():
    args = parse_args()
    dataset = DataPusher(root_path="data/kitech")
    detector = Detector(args, nms_thres = 0.5, score_thres2d = 0.7, score_thres3d = 0.1)
    tracker = Tracker()
    
    for i in tqdm(range(len(dataset))):
        # 1186 사람 첫 등장
        if i < 1186:
            continue
        rgb_image, depth_image, pcd = dataset.pusher(i)
        output2d, output3d = detector.inference(rgb_image, pcd)
        # # 사람, 자전거만 detecting
        if output2d is not None:
            print("2D detected")
        cur_tracks = tracker.matching(output2d, output3d, depth_image)
        
if __name__ == "__main__":
    main()