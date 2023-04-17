import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

img_types = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

parser = argparse.ArgumentParser(description="Function: convert images to video")
parser.add_argument('--input_dir', '-i', default="")
parser.add_argument('--output_dir', '-o', default="")
parser.add_argument('--output_name', '-n', default="")
parser.add_argument('--fps', '-f', type=float, default=25)
parser.add_argument('--resolution', '-r', type=int, nargs=2, default=[1280, 720])
parser.add_argument('--save_padding_img', '-s', default=False, action='store_true')
opt = parser.parse_args()

opt.input_dir = "D:\\Development\\Python\\Pycharm\\AI\\09GaitRecognitionSystem\\OpenGait\\OutdoorGait\\train\images\\001\\scene1_bg_L_090_1"
opt.output_dir = ""
opt.output_name = ""
opt.resolution = [320, 240]


# 将图片等比例缩放，不足则填充黑边
def resize_and_padding(img, target_size):
    size = img.shape
    h, w = size[0], size[1]
    target_h, target_w = target_size[1], target_size[0]

    # 确定缩放的尺寸
    scale_h, scale_w = float(h / target_h), float(w / target_w)
    scale = max(scale_h, scale_w)
    new_w, new_h = int(w / scale), int(h / scale)

    # 缩放后其中一条边和目标尺寸一致
    resize_img = cv2.resize(img, (new_w, new_h))

    # 图像上、下、左、右边界分别需要扩充的像素数目
    top = int((target_h - new_h) / 2)
    bottom = target_h - new_h - top
    left = int((target_w - new_w) / 2)
    right = target_w - new_w - left

    # 填充至 target_w * target_h
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return pad_img


def imgs2video(imgs_path, output_dir, target_name, target_size, target_fps, save_padding_img=False):
    if imgs_path:
        if not os.path.isdir(imgs_path):
            print("input is not a directory")
            sys.exit(0)

    if target_fps:
        if not target_fps > 0:
            print('fps should be greater than zero')
            sys.exit(0)

    if target_size:
        if not target_size[0] > 0 and target_size[1] > 0:
            print('resolution should be greater than zero')
            sys.exit(0)

    output_path = output_dir if output_dir else os.path.sep.join([imgs_path, "out"])
    os.makedirs(output_path, exist_ok=True)
    target = os.path.sep.join([output_path, target_name if target_name else "out.mp4"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(target, fourcc, target_fps, target_size)
    images = os.listdir(imgs_path)
    images.sort()
    count = 0
    for frame_name in images:
        if not (frame_name.lower().endswith(img_types)):
            continue

        try:
            # print(image)
            frame_path = os.path.sep.join([imgs_path, frame_name])
            # frame = cv2.imread(frame_path)  # imread 不能读中文路径，unicode也不行
            frame = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # , cv2.IMREAD_UNCHANGED
            pad_frame = resize_and_padding(frame, target_size)
            # print(pad_frame.shape)

            if save_padding_img:
                # 保存缩放填充后的图片
                resize_path = os.path.sep.join([output_dir, "resize"]) if output_dir else os.path.sep.join(
                    [imgs_path, "resize"])
                os.makedirs(resize_path, exist_ok=True)
                resize_name = os.path.sep.join([resize_path, "resize_" + frame_name])
                # cv2.imwrite(resize_name, pad_frame)  # imwrite 不能读中文路径，unicode也不行
                cv2.imencode(os.path.splitext(frame_name)[-1], pad_frame)[1].tofile(resize_name)

            # 写入视频
            vw.write(pad_frame)
            count += 1

        except Exception as exc:
            print(frame, exc)

    vw.release()
    print('\r\nConvert Success! Total ' + str(count) + ' images be combined into the video at: ' + target + '\r\n')


def main():
    imgs_path = opt.input_dir
    print('input path: ' + imgs_path)
    output_dir = opt.output_dir
    target_name = opt.output_name
    print('output dir:', output_dir + '/' + target_name)
    target_fps = opt.fps
    print('output file fps: ' + str(target_fps))
    target_size = (opt.resolution[0], opt.resolution[1])
    print('output file resolution: ' + str(target_size))
    save_padding_img = opt.save_padding_img

    imgs2video(imgs_path, output_dir, target_name, target_size, target_fps, save_padding_img)


def datasets_ext_video():
    INPUT_PATH = "OutdoorGait/train/images"
    OUTPUT_PATH = "OutdoorGait-video/train/videos"

    print(f"Walk to {INPUT_PATH}")

    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _id in tqdm(id_list):
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        for _seq_type in seq_type:
            imgs_path = os.path.join(INPUT_PATH, _id, _seq_type)
            video_output_path = os.path.join(OUTPUT_PATH, _id)
            target_name = _id + "_" + _seq_type + ".mp4"
            target_size = [320, 240]
            target_fps = 25
            print(f"{imgs_path=}")
            print(f"{target_name=}, {target_size=}, {target_fps=}")
            imgs2video(imgs_path,video_output_path, target_name, target_size, target_fps)


if __name__ == '__main__':
    # main()
    datasets_ext_video()
