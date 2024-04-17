import sys
import numpy as np
import cv2
import glob
from pathlib import Path
import os
from tqdm import tqdm
data_img = "data/test/camera/*.png"
rst_img_dir = "work_dirs/test"
video_name = "test_1.avi"
H = 720
W = 1280
def image_load(data_img, rst_img_dir, H, W):

    img_array = []

    for filename in tqdm(sorted(glob.glob(data_img))):
        final_img = np.zeros((H*2, W, 3), dtype=np.uint8)
        result_img = Path(filename).name
        result_img = str(Path(rst_img_dir) / result_img)
        img = cv2.imread(filename)
        img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        result_img = cv2.imread(result_img)
        
        final_img[:H,:,:] = img
        final_img[H:,:,:] = result_img
        size = (2*H, W)
        
        img_array.append(final_img)

    return size, img_array


def video_generator(folder_path, file_name, size, fps, img_array):

    out = cv2.VideoWriter(filename=os.path.join(folder_path, file_name), fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=fps, frameSize= (W, 2*H))

    for img in img_array:
        out.write(img)

    out.release()


def video_play(file_name):

    cap = cv2.VideoCapture(file_name)

    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_size, img_array = image_load(data_img, rst_img_dir, H, W)
    video_generator(folder_path= rst_img_dir, file_name=video_name, size=img_size, fps=15, img_array=img_array)
    video_play(file_name=video_name)