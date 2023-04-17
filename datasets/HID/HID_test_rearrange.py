import os
import shutil
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--HID_dataset_path', default='HID2022-my-test2', type=str, help='Rearrange HID datasets to probe.')

opt = parser.parse_args()

WORK_PATH = "."
os.chdir(WORK_PATH)
print("WORK_PATH:", os.getcwd())


def rearrange_hid_to_probe_test():
    INPUT_PATH = opt.HID_dataset_path

    print(f"Walk to {INPUT_PATH}")

    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _id in tqdm(id_list):
        if _id == "probe" or int(_id) >= 500:
            continue
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        length = len(seq_type)

        probe_seq_type = random.sample(seq_type, int(0.6 * length))

        for _seq_type in probe_seq_type:
            src_path = os.path.join(INPUT_PATH, _id, _seq_type)
            dst_path = os.path.join(INPUT_PATH, "probe", _id, _seq_type)

            default_path = os.path.join(src_path, "default")
            pkl_path = os.path.join(src_path, "default", str(_seq_type) + ".pkl")

            dst_pkl_path = os.path.join(src_path, str(_seq_type) + ".pkl")

            if os.path.exists(pkl_path):
                shutil.move(pkl_path, dst_pkl_path)
                os.rmdir(default_path)
                shutil.move(src_path, dst_path)


def rearrange_hid_to_common_test():
    INPUT_PATH = opt.HID_dataset_path

    print(f"Walk to {INPUT_PATH}")

    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _id in tqdm(id_list):
        if _id == "probe" or int(_id) >= 500:
            continue
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        length = len(seq_type)

        probe_seq_type = random.sample(seq_type, int(0.6 * length))
        gallery_seq_type = list(set(seq_type).difference(set(probe_seq_type)))
        print(probe_seq_type)
        print(gallery_seq_type)

        for _seq_type in seq_type:
            src_path = os.path.join(INPUT_PATH, _id, _seq_type)
            # move pkl, rm default folder
            default_path = os.path.join(src_path, "default")
            pkl_path = os.path.join(src_path, "default", str(_seq_type) + ".pkl")
            dst_pkl_path = os.path.join(src_path, str(_seq_type) + ".pkl")
            # partition 00 and 01
            probe_path = os.path.join(INPUT_PATH, _id, "00")
            gallery_path = os.path.join(INPUT_PATH, _id, "01")
            os.makedirs(probe_path, exist_ok=True)
            os.makedirs(gallery_path, exist_ok=True)

            if os.path.exists(pkl_path):
                shutil.move(pkl_path, dst_pkl_path)
                os.rmdir(default_path)
                if _seq_type in probe_seq_type:
                    shutil.move(src_path, probe_path)
                else:
                    shutil.move(src_path, gallery_path)


if __name__ == '__main__':
    # rearrange_hid_to_probe_test()
    rearrange_hid_to_common_test()







