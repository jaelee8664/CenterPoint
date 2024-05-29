import numpy as np
import pickle
import PIL
import os
from tqdm import tqdm
from mot_lidar_rgbd import DataPusher
import matplotlib.pyplot as plt
def main():
    with open("mot_result.pickle", "rb") as f:
        rst = pickle.load(f)
    dataset = DataPusher(root_path="data/kitech")
    work_dir = "work_dirs/final_mot_result"
    os.makedirs(work_dir, exist_ok=True)
    work_dir1 = "work_dirs/kitech_data_mot"
    work_dir2 = "work_dirs/kitech_data_mot2"
    for i in tqdm(range(len(rst))):
        a = work_dir1 + "/" + str(rst[i]["frame_num"]) +".png"
        b = work_dir2 + "/" + str(rst[i]["frame_num"]) +".png"
        a = np.array(PIL.Image.open(a), dtype=np.uint8)
        b = np.array(PIL.Image.open(b), dtype=np.uint8)
        rgb_image, depth_image, _ = dataset.pusher(rst[i]["frame_num"])
        fig = plt.figure(figsize=(20,17))
        ax1 = fig.add_axes([0.05, 0.31, 0.45, 0.8])
        ax1.imshow(a)
        ax1.set_title('ID Box')
        ax1.axis("off")
        
        ax2 = fig.add_axes([0.54, 0.31, 0.45, 0.8])
        ax2.imshow(b)
        ax2.set_title('Pred(green) Box + Measurement(red) Box')
        ax2.axis("off")
        # ax3 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_axes([0.01, 0, 0.45, 0.3])
        ax3.imshow(rgb_image)
        ax3.set_title('color image')
        # ax3.imshow(img3)
        # ax3.set_title('projected points and boxes')
        ax3.axis("off")

        # ax4 = fig.add_subplot(2, 2, 4)
        ax4 = fig.add_axes([0.54, 0, 0.45, 0.3])
        ax4.imshow(depth_image)
        ax4.set_title('depth image')
        # ax4.imshow(img4)
        # ax4.set_title('2D boxes and projected boxes')
        ax4.axis("off")
    
        fig.savefig(work_dir + "/" + str(rst[i]["frame_num"]) +".png")
        plt.cla()
        plt.clf()
        plt.close(fig)
            
if __name__ == "__main__":
    main()
    