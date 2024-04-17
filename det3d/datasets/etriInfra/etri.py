import pickle
from pathlib import Path

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
)
from det3d.datasets.registry import DATASETS
import os
@DATASETS.register_module
class etrInfraDataset(PointCloudDataset):
    NumPointFeatures = 3
    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="",
        load_interval=1,
        **kwargs,
    ):
        
        self.load_interval = load_interval 
        super(etrInfraDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )
        
        self.nsweeps = nsweeps
        self._info_path = info_path
        self._class_names = class_names
        
        with open(info_path, "rb") as f:
            self._nusc_infos = pickle.load(f)
        
        self._num_point_features = etrInfraDataset.NumPointFeatures
        self._name_mapping = general_to_detection
        
        self.virtual = kwargs.get('virtual', False)
        if self.virtual:
            self._num_point_features = 16 
            
        self.version = version
        self.eval_version = "detection_cvpr_2019"
        
    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            with open(self._info_path, "rb") as f:
                self._nusc_infos = pickle.load(f)

        return len(self._nusc_infos)
    
    def get_sensor_data(self, idx):

        info = self._nusc_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": 1,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": Path("data/etri3Dobj_infra_edge"),
                "num_point_features": self._num_point_features,
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train"
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)
    
    
@DATASETS.register_module
class TestDataset(PointCloudDataset):
    NumPointFeatures = 3
    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="",
        load_interval=1,
        **kwargs,
    ):
        
        self.load_interval = load_interval 
        super(TestDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )
        self.nsweeps = nsweeps
        self._info_path = info_path
        self._class_names = class_names
        
        self._num_point_features = TestDataset.NumPointFeatures
        self._name_mapping = general_to_detection
        
        self.virtual = kwargs.get('virtual', False)
        if self.virtual:
            self._num_point_features = 16 
            
        self.version = version
        
        # for (root, dirs, file) in os.walk(str(self._root_path)):
        #     for f in file:    
        #         file_name = str(Path(root)/Path(f))
        #         self.file_dirs.append(file_name)
        
        
    def __len__(self):
        if not hasattr(self, "file_dirs"):
            
            self.pcd_file_name = []
            for (root, dirs, file) in os.walk(str(self._root_path)):
                if root == str(self._root_path) + "/pcd":
                    for f in file:
                        self.pcd_file_name.append(Path(f))

            # self.file_dirs = []
            # for (root, dirs, file) in os.walk(str(self._root_path)):
            #     for f in file:    
            #         self.file_dirs.append(Path(f))

        
        return len(self.pcd_file_name)
    
    def get_sensor_data(self, idx):

        info = self.pcd_file_name[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": 1,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "data_root" : self._root_path,
                "image_prefix": Path("camera"),
                "pcd_prefix": Path("pcd"),
                "num_point_features": self._num_point_features,
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train"
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)