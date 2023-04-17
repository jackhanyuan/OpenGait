import os
import cv2
import time
import numpy as np
import argparse
import pickle

from PIL import Image
from tqdm import tqdm
from is_people.classification import Classification
classfication = Classification()

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_train_path', default='HID2022/HID2022_train', type=str,
                    help='Root path of train.')
parser.add_argument('--input_gallery_path', default='HID2022/HID2022_test_gallery', type=str,
                    help='Root path of gallery.')
parser.add_argument('--input_probe_path', default='HID2022/HID2022_test_probe', type=str,
                    help='Root path of probe.')
parser.add_argument('--output_path', default='HID2022-clean-augment-pkl-64', type=str,
                    help='Root path for output.')
parser.add_argument('--img_size', default=64, type=int, help='Image resizing size. Default 64')
parser.add_argument('--augment', default=True, type=bool, help='Image Horizontal Flip Augmented Dataset')
parser.add_argument('--is_people', default=True, type=bool, help='judge is people')

opt = parser.parse_args()
OUTPUT_PATH = opt.output_path

print('Pretreatment Start.\n'
      'Input train path: {}\n'
      'Input gallery path: {}\n'
      'Input probe path: {}\n'
      'Output path: {}\n'.format(
          opt.input_train_path, opt.input_gallery_path, opt.input_probe_path, OUTPUT_PATH))

WORK_PATH = "./"
os.chdir(WORK_PATH)
print("WORK_PATH:", os.getcwd())
print(f"{opt.img_size=}")


def is_people(img):
    if opt.is_people:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        class_name, probability = classfication.detect_image(img)
        if class_name == 'people' and probability > 0.5:
            return True
        else:
            return False
    return True


def imgs_to_pickle(_id, _seq_type, INPUT_PATH, OUTPUT_PATH):
    out_dir = os.path.join(OUTPUT_PATH, _id, _seq_type, "default")
    all_imgs_pkl = os.path.join(out_dir, '{}.pkl'.format(_seq_type))
    if os.path.exists(all_imgs_pkl):
        print('\t Exists:', all_imgs_pkl)
        return
    count_frame = 0
    all_imgs = []
    flip_imgs = []
    base_dir = os.path.join(INPUT_PATH, _seq_type) if _id == 'probe' else os.path.join(INPUT_PATH, _id, _seq_type)
    frame_list = sorted(os.listdir(base_dir))
    for frame_name in frame_list:
        if frame_name.lower().endswith(('png', 'jpg')):  # filter png files
            frame_path = os.path.join(base_dir, frame_name)
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print('\t RM:', frame_path)
                os.remove(frame_path)
                continue
                
            if is_people(img):
                # resize
                ratio = img.shape[1] / img.shape[0]
                img = cv2.resize(img, (int(opt.img_size * ratio), opt.img_size), interpolation=cv2.INTER_CUBIC)
                
                # Save the img
                all_imgs.append(img)
                count_frame += 1
                
                # img = np.array(img)
                # cv2.imwrite('people/true/' + str(time.perf_counter()) + '.png', img)

                if opt.augment:
                    flip_img = cv2.flip(img, 1)  # 水平翻转，扩充数据
                    flip_imgs.append(flip_img)
                    print('\t augment:', frame_path)

            else:
                print('\t no people:', frame_path)
                # print('\t RM:', frame_path)
                # os.remove(frame_path)
                # img = np.array(img)
                # cv2.imwrite('people/false/' + str(time.perf_counter()) + '.png', img)

    all_imgs = np.asarray(all_imgs + flip_imgs)

    if count_frame > 0:
        os.makedirs(out_dir, exist_ok=True)
        all_imgs_pkl = os.path.join(out_dir, '{}.pkl'.format(_seq_type))
        pickle.dump(all_imgs, open(all_imgs_pkl, 'wb'))

    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        print('Seq:{}-{}, less than 5 valid data.'.format(_id, _seq_type))

    return


if __name__ == '__main__':

    INPUT_PATH = opt.input_train_path

    print("Walk the input train path")
    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _id in tqdm(id_list):
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        for _seq_type in seq_type:
            imgs_to_pickle(_id, _seq_type, INPUT_PATH, OUTPUT_PATH)

    print("Walk the input gallery path")
    INPUT_PATH = opt.input_gallery_path
    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _id in tqdm(id_list):
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        for _seq_type in seq_type:
            imgs_to_pickle(_id, _seq_type, INPUT_PATH, OUTPUT_PATH)

    print("Walk the input probe path")
    INPUT_PATH = opt.input_probe_path
    seq_type = os.listdir(INPUT_PATH)
    seq_type.sort()
    _id = "probe"
    for _seq_type in tqdm(seq_type):
        imgs_to_pickle(_id, _seq_type, INPUT_PATH, OUTPUT_PATH)
