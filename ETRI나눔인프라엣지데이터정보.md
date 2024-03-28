# ETRI 나눔 인프라 엣지 3D 동적객체 검출/추적 학습데이터
출처: https://nanum.etri.re.kr/share/kakao_admin/kakaomobilityInfraedge3DObjectDetection?lang=ko_KR

## 총 19개의 데이터 폴더로 구성

1: 'round(20220406153541_Teslasystem_PE01)_time(1649226941_1649226984)'
<br/> 
2: 'round(20220525184544_Teslasystem_PE01)_time(1653471944_1653471987)'
<br/> 
\.\.\.\.
<br/> 
19: 'round(20220825100350_Teslasystem_PE01)_time(1661421830_1661422130)'

## 각 데이터 폴더 안에 meta, sensor 폴더로 구성
### meta: 메타데이터
1. dataset.json: 데이터 정보
    * uuid: 데이터 폴더 id
    * log_uuid: 로그 id(?)
    * scenario_names: 환경 (["day", "clear", "free_flow", "urban_road"])
    * name: 해당하는 데이터 폴더 이름
    * description: 데이터셋을 확보한 장소, 시간(어떤 시간인지 불분명) 정보
    * num_frames: 프레임 개수
    * first_frame_uuid: 첫 프레임 id
    * last_frame_uuid: 마지막 프레임 id
    * base_path: 베이스 폴더 이름 ("/3D_object_detection_tracking(edge_infra)/")
  
2. ego_pose.json: 차량 자세 정보(global 좌표계가 차량 자세 정보이다.)
    * uuid: 프레임(?) id
    * timestamp: 시간
    * translation: global기준 ego 좌표 [0, 0, 0]
    * rotation: global기준 ego좌표 [1, 0, 0, 0]
    * coordinate_system: null 또는  {"epsg": 32652, "ellipsoid": "wgs84", "geoid": ""}
3. frame_annotation.json
    * uuid: frame_annotaion의 id
    * frame_data_uuid: frame_data의 id
    * instance_uuid: 인스턴스 id
    * visibility_level: null(?)
    * annotation_type_name: bbox_pcd3d(?)
    * is_lidar_synced: 라이다 합성 여부(?)
    * geometry
        * wlh: 박스 너비 R^3
        * center: 중앙 좌표 R^3
        * orientation: 각도 R^4
    * num_pts: 포인트 개수(?)
    * attribute: 속성
        * vehicle_state: ("stopped" 등 )
        * human_state: ("standing" 등)
    * frame_aligned: ?
    * description : null
    * prev: 전 frame_annotation id
    * next: 다음 frame_annotation id
4. frame_data.json
    * uuid: frame_data의 id
    * frame_uuid: frame의 id
    * ego_pose_uuid: ego_pose의 id
    * sensor_uuid: 해당 센서의 id
    * timestamp: 시간
    * is_key_frame: true
    * height: 사진일 경우 높이
    * width: 사진일 겨우 너비
    * file_name: 센서 데이타 파일 이름
    * file_format: 센서 유형 (pcd, png)
    * prev
    * next
    * prev_key_frame_data
    * next_key_frame_data
5. frame.json
    * uuid: frame의 id
    * timestamp: 시간
    * dataset_uuid: 데이터 폴더 id
    * prev: 전 프레임 id (첫 프레임은 uuid와 같거나 null)
    * next: 다음 프레임 id (마지막 프레임은 uuid와 같거나 null)
6. instance.json
    * uuid: 인스턴스 id
    * category_name: 인스턴스 종류 ("dynamic_object.vehicle.car",ect)
    * first_frame_annotation_uuid: 인스턴스의 처음 등장한 frame_annotation의 id
    * last_frame_annotation_uuid: 인스턴스의 마지막 등장한 frame_annotation의 id
    * num_annotations: annotation된 횟수
    * description: null
7. log.json
    * uuid
    * map_uuids
    * device
    * date_captured
    * location
    * driving_distance
    * device_wlh
8. sensor.json
    [{"uuid": "9f979cf9-468f-4ea9-a518-47e87290c669", "name": "lidar(00)", "type": "lidar", "translation": [0, 0, 0], "rotation": [1, 0, 0, 0], "intrinsic": {"model": null, "parameter": null}, "height_from_ground": 1.6}, 
     {"uuid": "7092ac10-4e40-4a31-88c9-bc7d00054503", "name": "camera(00)", "type": "camera", "translation": [0.3959527732272198, 0.09038752100548936, 2.0665511303388615], "rotation": [0.29419437909735213, -0.4624695742918969, 0.6983071005061592, -0.4603680631239271], "intrinsic": {"model": "MODEL_MATRIX", "parameter": [[1056.123, 0, 1042.263], [0, 1060.503, 802.543], [0, 0, 1]]}, "height_from_ground": 3.8}, 
     {"uuid": "6ca55d5b-c7b9-433e-8ff5-63c9b7bbd96e", "name": "camera(01)", "type": "camera", "translation": [0.0012533625667005739, -0.2611764986620123, 2.0147914436818652], "rotation": [-0.02779685235033416, 0.038208678783245666, 0.79472107791341, -0.6051329111734052], "intrinsic": {"model": "MODEL_MATRIX", "parameter": [[1058.411, 0, 1025.369], [0, 1062.198, 752.831], [0, 0, 1]]}, "height_from_ground": 3.8}, 
     {"uuid": "6eab30a9-797c-4690-b84d-db195e1fc6f6", "name": "camera(02)", "type": "camera", "translation": [-0.3569543846651315, -0.0129641575802949, 2.015169395387349], "rotation": [-0.1877969807216212, 0.254657814358064, 0.7882939983671278, -0.527706607649159], "intrinsic": {"model": "MODEL_MATRIX", "parameter": [[1057.518, 0, 1013.904], [0, 1063.306, 782.263], [0, 0, 1]]}, "height_from_ground": 3.8}]

### sensor: 센서 데이터
1. camera(00)<br/> 
2. camera(01)<br/> 
3. camera(02)<br/> 
4. lidar(00): pcd 파일