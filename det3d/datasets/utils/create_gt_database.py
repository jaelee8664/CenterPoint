import pickle
from pathlib import Path
import os 
import numpy as np

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm

# etriInfra에 필요한 라이브러리
import json
import open3d as o3d

# class 이름 간소화 딕셔너리
instance_to_name = {'dynamic_object.vehicle.personal_mobility': 'personal_mobility',
                    'dynamic_object.vehicle.bus': 'bus',
                    'dynamic_object.vehicle.bicycle': 'bicycle',
                    'dynamic_object.vehicle.car' : 'car',
                    'dynamic_object.vehicle.construction_vehicle': 'construction_vehicle',
                    'dynamic_object.vehicle.motorcycle': 'motorcycle',
                    'dynamic_object.vehicle.truck': 'truck',
                    'dynamic_object.human.pedestrian': 'pedestrian',
                    'dynamic_object.animal.ground_animal': 'ground_animal'}

dataset_name_map = {
    "NUSC": "NuScenesDataset",
    "WAYMO": "WaymoDataset"
}

def quaternion_to_euler_angle_z(orientation):
    """_summary_

    Args:
        orientation (_type_): 쿼터니언 형식의 orientation

    Returns:
        _type_: euler z각도 
    """
    w, x, y, z = orientation
    ysqr = y * y

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    # Z = np.degrees(np.arctan2(t3, t4))
    Z = np.arctan2(t3, t4)
    return Z

def create_etri_groundtruth_databse(
    root_dir
):
    """_summary_
    Args:
        root_dir (_type_): etri 데이터 root 폴더 경로
    Description: 1. 각 인스턴스(물체)가 갖는 포인트 클라우드 데이터 .bin 확장자로 gt_database_withoutvelo폴더에 저장 (데이터 augmentation에 필요) => gt_points
                 2. 각 인스턴스의 정보를 담은 pickle 데이터 저장 (데이터 augmentation에 필요) => all_db_infos
                 3. DataLoader에 올리기 용이하도록 설계된 gt 전처리 데이터 pickle 데이터 저장 => train_list, val_list
    """
    data_folder = [] # 19개 장면의 데이터 폴더 경로
    data = dict() # 데이터 placeholder
    meta_file= ['sensor.json', 'log.json', 'dataset.json', 'ego_pose.json', 
                'frame_annotation.json', 'instance.json', 'frame_data.json', 'frame.json'] # 메타데이터 파일
    frame_data_to_frame_uuid = dict() #frame_data.json Id에서 frame.json Id로 변환하는 딕셔너리
    instance_to_category_name = dict() #instance.json Id에서 클래스 이름으로 변환하는 딕셔너리
    id_to_key = dict() # dataset.json id에서 숫자 인덱스로 변환하는 딕셔너리
    
    train_list = [] # 훈련 데이터
    val_list = [] # 검증 데이터
    
    
    root_path = Path(root_dir)
    dbinfo_path = root_path / f"db_meta_withoutvelo.pkl" # 인스턴스 포인트 meta 데이터
    db_path = root_path / f"db_database_withoutvelo" # 인스턴스 포인트 데이터 저장 폴더 경로
    
    # data_folder 업데이트
    for (root, dirs, file) in os.walk(root_dir):
        if "meta" in dirs:
            data_folder.append(root)
    
    # placeholder 생성
    for d in data_folder:
        for m in meta_file:
            file_name = d + '/meta/' + m
            if m == 'frame.json':
                with open(file_name, 'r') as f:
                    json_data = json.load(f)
                idx = 0

                nex_frame_uuid = json_data[idx]['uuid']
                while (nex_frame_uuid != None):
                    cur_frame = json_data[idx]
                    assert (nex_frame_uuid == cur_frame['uuid'])
                    nex_frame_uuid = cur_frame['next']
                    
                    if cur_frame['dataset_uuid'] not in data:
                        data[cur_frame['dataset_uuid']] = dict()
                    # data -> dataset_uuid -> frame_uuid 순으로 디렉토리 생성
                    data[cur_frame['dataset_uuid']][cur_frame['uuid']] = dict()
                    # det3d 인풋형식 맞추기
                    data[cur_frame['dataset_uuid']][cur_frame['uuid']]['type'] = 'lidar'
                    idx += 1
    
    # placeholder 업데이트
    for d in tqdm(data_folder):
        for m in meta_file:
            file_name = d + '/meta/' + m
            if m == 'frame_data.json':
                with open(file_name, 'r') as f:
                    json_data = json.load(f)
                for i in json_data:
                    if (i['file_format'] == 'pcd'):
                        lidar_file_name = d + '/sensor/lidar(00)/' + i['file_name'] + '.' + i['file_format']
                        pcd = o3d.io.read_point_cloud(lidar_file_name)
                        out_arr = np.asarray(pcd.points, dtype=np.float32)  
                        for k in data.keys():
                            if i['frame_uuid'] in data[k]:
                                data[k][i['frame_uuid']]['points'] = out_arr
                                data[k][i['frame_uuid']]['annotations'] = {'boxes': [], 'names': []}
    
    # frame_data_to_frame_uuid 업데이트
    for d in data_folder:
        for m in meta_file:
            file_name = d + '/meta/' + m
            if m == 'frame_data.json':
                with open(file_name, 'r') as f:
                    json_data = json.load(f)
                for i in json_data:
                    frame_data_to_frame_uuid[i['uuid']] = i['frame_uuid']
    
    # instance_to_category_name 업데이트
    for d in data_folder:
        for m in meta_file:
            file_name = d + '/meta/' + m
            if m == 'instance.json':
                with open(file_name, 'r') as f:
                    json_data = json.load(f)
                for i in json_data:
                    instance_to_category_name[i['uuid']] = instance_to_name[i['category_name']]
    
    # gt 데이터 전처리
    print("Making GT data Pkl")
    for d in tqdm(data_folder):
        for m in meta_file:
            file_name = d + '/meta/' + m
            if m == 'frame_annotation.json':
                with open(file_name, 'r') as f:
                    json_data = json.load(f)
                for i in json_data:
                    if i['is_lidar_synced'] == False:
                        try:
                            tmp_arr = np.concatenate((np.array(i['geometry']['center']), np.array(i['geometry']['wlh']), np.array([0.0, 0.0, quaternion_to_euler_angle_z(i['geometry']['orientation'])])))
                        except:
                            continue
                        # print(frame_data_to_frame_uuid[i['frame_data_uuid']])
                        frame_uuid = frame_data_to_frame_uuid[i['frame_data_uuid']]
                        for k in data.keys():
                            try:
                                if frame_uuid in data[k]:
                                    data[k][frame_uuid]['annotations']['boxes'].append(tmp_arr)
                                    data[k][frame_uuid]['annotations']['names'].append(instance_to_category_name[i['instance_uuid']])
                            except:
                                print("Creating GT Data is not completed")
            
    # 훈련, 검증 전처리 데이터 생성
    i = 0
    for k in data.keys():
        id_to_key[i] = k
        i += 1
    for i in range(len(data) -1):
        idx = id_to_key[i]
        scene_data = data[idx]  
        for k in scene_data.keys():
            data[idx][k]['num_scene'] = i + 1
            data[idx][k]['annotations']['boxes'] = np.stack(scene_data[k]['annotations']['boxes'], axis = 0)
            data[idx][k]['annotations']['names'] = np.array(scene_data[k]['annotations']['names'])
            train_list.append(data[idx][k])
    val_i = 18
    val_idx = id_to_key[val_i]
    scene_data = data[val_idx]
    for k in scene_data.keys():
        data[val_idx][k]['num_scene'] = val_i + 1
        data[val_idx][k]['annotations']['boxes'] = np.stack(scene_data[k]['annotations']['boxes'], axis = 0)
        data[val_idx][k]['annotations']['names'] = np.array(scene_data[k]['annotations']['names'])
        val_list.append(data[val_idx][k])
        
    print(f"len(train_list): {len(train_list)}, len(val_list): {len(val_list)}")
    
    # 저장
    with open(root_path / "infos_train.pkl", "wb") as f:
        pickle.dump(train_list, f)
    with open(root_path / "infos_val.pkl", "wb") as f:
        pickle.dump(val_list, f)
    
    # 인스턴스 포인트 제작 및 저장
    all_db_infos = {}
    group_counter = 0
    print("Making DB data Pkl")
    for index in tqdm(range(len(train_list))):
        group_dict = {}
        num_obj = train_list[index]["annotations"]["boxes"].shape[0]
        points = train_list[index]["points"]
        gt_boxes = train_list[index]["annotations"]["boxes"]
        names = train_list[index]["annotations"]["names"]
        group_ids = np.arange(train_list[index]["annotations"]["boxes"].shape[0], dtype=np.int64)
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        if num_obj == 0:
            continue 
        
        # 인스턴스가 갖는 포인트 추출
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        # point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            filename = f"{index}_{names[i]}_{i}.bin"
            dirpath = os.path.join(str(db_path), names[i])
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(str(db_path), names[i], filename)
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, "wb") as f:
                try:
                    f.write(bytes(gt_points))
                    # gt_points.write(f)
                except:
                    print("process {} files".format(index))
                    break
            db_dump_path = str(filepath)
            db_info = {
                "name": names[i],
                "path": db_dump_path,
                "image_idx": index,
                "gt_idx": i,
                "box3d_lidar": gt_boxes[i],
                "num_points_in_gt": gt_points.shape[0],
                "difficulty": difficulty[i],
                # "group_id": -1,
                # "bbox": bboxes[i],
            }
            local_group_id = group_ids[i]
            if local_group_id not in group_dict:
                group_dict[local_group_id] = group_counter
                group_counter += 1
            db_info["group_id"] = group_dict[local_group_id]
            
            if names[i] in all_db_infos:
                all_db_infos[names[i]].append(db_info)
            else:
                all_db_infos[names[i]] = [db_info]
    # 저장
    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)

def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    virtual=False,
    **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            nsweeps=kwargs["nsweeps"],
            virtual=virtual
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name in ["WAYMO", "NUSC"]: 
        if db_path is None:
            if virtual:
                db_path = root_path / f"gt_database_{nsweeps}sweeps_withvelo_virtual"
            else:
                db_path = root_path / f"gt_database_{nsweeps}sweeps_withvelo"
        if dbinfo_path is None:
            if virtual:
                dbinfo_path = root_path / f"dbinfos_train_{nsweeps}sweeps_withvelo_virtual.pkl"
            else:
                dbinfo_path = root_path / f"dbinfos_train_{nsweeps}sweeps_withvelo.pkl"
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        if nsweeps > 1: 
            points = sensor_data["lidar"]["combined"]
        else:
            points = sensor_data["lidar"]["points"]
            
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]

        if dataset_class_name == 'WAYMO':
            # waymo dataset contains millions of objects and it is not possible to store
            # all of them into a single folder
            # we randomly sample a few objects for gt augmentation
            # We keep all cyclist as they are rare 
            if index % 4 != 0:
                mask = (names == 'VEHICLE') 
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

            if index % 2 != 0:
                mask = (names == 'PEDESTRIAN')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue 
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points.tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)
