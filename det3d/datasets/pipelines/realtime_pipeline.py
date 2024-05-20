import numpy as np
import torch
from ..registry import PIPELINES
from det3d.core.input.voxel_generator import VoxelGenerator
import collections

@PIPELINES.register_module
class PreprocessRealtime():
    def __init__(self, device, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num
        self.device = device
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )
        self.device = device
    def __call__(self, points):
        grid_size = self.voxel_generator.grid_size
        
        max_voxels = self.max_voxel_num[1]
        
        voxels, coordinates, num_points = self.voxel_generator.generate(
            points, max_voxels=max_voxels 
        )
        # voxels = torch.from_numpy(voxels)
        # points = torch.from_numpy(points)
        # num_points = torch.from_numpy(num_points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        # num_voxels = torch.from_numpy(num_voxels)
        # coordinates = torch.from_numpy(coordinates)
        data_bundle = dict()
        if points is not None:
            data_bundle.update(
                points=points,
                voxels=voxels,
                shape=grid_size,
                num_points=num_points,
                num_voxels=num_voxels,
                coordinates=coordinates
            )
        
            data_bundle = collate(data_bundle)
        return data_bundle
    
def collate(example, samples_per_gpu=1):
    example_merged = collections.defaultdict(list)

    for k, v in example.items():
        example_merged[k].append(v)
    batch_size = len(example_merged['num_voxels'])
    ret = {}
    # voxel_nums_list = example_merged["num_voxels"]
    # example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels",
                   "cyv_voxels", "cyv_num_points", "cyv_num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))
        elif key in [
            "gt_boxes",
        ]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key == "points":
            ret[key] = [torch.tensor(elem) for elem in elems]
        elif key in ["coordinates", "cyv_coordinates"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_box",
                    "ind", "mask", "cat"]:

            ret[key] = collections.defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            for kk, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res
        elif key == 'gt_boxes_and_cls':
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret
