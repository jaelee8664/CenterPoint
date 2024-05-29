import pickle
import os
import numpy as np
import open3d as o3d
import PIL


def get_color(idx):
    """
    choose color for plotting box
    """
    idx = idx * 3
    color = [((37 * idx) % 255)/255, ((17 * idx) % 255)/255, ((29 * idx) % 255)/255]
    return color

def main():
    with open("mot_result.pickle", "rb") as f:
        rst = pickle.load(f)
    vis = o3d.visualization.Visualizer()
    uniform_color = [0, 0, 1]  # Blue color, you can change it to any other color you like
    vis.create_window(width=1280, height=720)  # Set the window size as per your requirement
    vis.get_render_option().point_size = 1.0  # Set point size
    vis.get_render_option().line_width = 10

    # Set camera parameters
    ctr = vis.get_view_control()
    work_dir = "work_dirs/kitech_data_mot"
    os.makedirs(work_dir, exist_ok=True)
    parameters = o3d.io.read_pinhole_camera_parameters("tools/vis_utils/ScreenCamera_2024-04-30-09-53-54.json")
    for i in range(len(rst)):
        """
        rst.append({
            "frame_num" : i,
            "pcd_file": pcd_file,
            "rgb_file": rgb_file,
            "depth_file": depth_file,
            "cur_tracks" : cur_tracks
        })
        """
        # 포인트 그리기
        pcd = o3d.io.read_point_cloud(rst[i]["pcd_file"])
        points = np.asarray(pcd.points, dtype=np.float32)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # img = np.array(PIL.Image.open(rst[i]["rgb_file"]), dtype=np.uint8)
        # depth_raw = np.array(PIL.Image.open(rst[i]["depth_file"]), dtype=np.uint16)
        # depth = depth_raw / 200
        
        pcd_col = np.tile(uniform_color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(pcd_col)  # Set uniform color for all points
        vis.clear_geometries()
        vis.add_geometry(pcd)
        
        # 박스 그리기
        track_boxes = rst[i]["cur_tracks"]
        pred_boxes = rst[i]["pred_tracks"]
        detection_boxes = rst[i]["output2d"]
        for j in range(len(track_boxes)):
            w = l = h = 1
            bounding_box = np.array([
                            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
            id = track_boxes[j][0]
            id_text = '{}'.format(int(id))
            color = get_color(abs(id))
            translation = track_boxes[j][1:4]
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
            box.color = color           #Box color would be red box.color = [R,G,B]
            vis.add_geometry(box)
        # for j in range(len(pred_boxes)):
        #     w = l = h = 1
        #     bounding_box = np.array([
        #                     [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        #                     [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        #                     [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
        #     color = [0, 1, 0]
        #     translation = pred_boxes[j]
        #     eight_points = np.tile(translation, (8, 1))
        #     rotation = np.array([[ 0.59587538,  0.80307692,  0.   ],
        #                     [-0.80307692,  0.59587538,  0.        ],
        #                     [ 0.,          0.,          1.        ]])
        #     corner_box = np.dot(rotation, bounding_box) + eight_points.transpose()
        #     boxes3d_pts = corner_box.transpose()
        #     # boxes3d_pts[:, 2] = np.clip(boxes3d_pts[:, 2], -1.6, 60) # 박스의 아랫부분을 잘라내기 위한 클리핑
        #     boxes3d_pts = boxes3d_pts.T
            
        #     boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts.T)
        #     # boxes3d_pts_list.append(np.asarray(boxes3d_pts))
        #     box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        #     box.color = color           #Box color would be red box.color = [R,G,B]
        #     vis.add_geometry(box)
        # if detection_boxes is not None:
        #     for j in range(len(detection_boxes)):
        #         w = l = h = 1
        #         bounding_box = np.array([
        #                         [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        #                         [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        #                         [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
        #         color = [1, 0, 0]
        #         translation = detection_boxes[j, :3]
        #         eight_points = np.tile(translation, (8, 1))
        #         rotation = np.array([[ 0.59587538,  0.80307692,  0.   ],
        #                         [-0.80307692,  0.59587538,  0.        ],
        #                         [ 0.,          0.,          1.        ]])
        #         corner_box = np.dot(rotation, bounding_box) + eight_points.transpose()
        #         boxes3d_pts = corner_box.transpose()
        #         # boxes3d_pts[:, 2] = np.clip(boxes3d_pts[:, 2], -1.6, 60) # 박스의 아랫부분을 잘라내기 위한 클리핑
        #         boxes3d_pts = boxes3d_pts.T
                
        #         boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts.T)
        #         # boxes3d_pts_list.append(np.asarray(boxes3d_pts))
        #         box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        #         box.color = color           #Box color would be red box.color = [R,G,B]
        #         vis.add_geometry(box)
        # 캡쳐
        vis.poll_events()
        ctr.convert_from_pinhole_camera_parameters(parameters, True)
        vis.update_renderer()
        vis.run()
        a = work_dir + "/" + str(rst[i]["frame_num"]) +".png"
        print(a)
        vis.capture_screen_image(a)
        
if __name__ == "__main__":
    main()