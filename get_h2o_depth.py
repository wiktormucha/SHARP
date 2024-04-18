from PIL import Image
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from spock_dataclasses import *
from tqdm import tqdm
import os
from datasets.h2o import H2O_Dataset_hand_train_3D

device = 'cuda:0'
MAX_NUM_THREADS = 16
torch.set_num_threads(MAX_NUM_THREADS)


def get_pth_for_h2o_depth(pth_rgb, depth_folder='depth_est_low'):
    return pth_rgb.replace('rgb', depth_folder)


def main():
    model_rgb2depth = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas").to(device)
    feature_extractor = DPTFeatureExtractor.from_pretrained(
        "Intel/dpt-hybrid-midas")

    spock_cfg = H2oHandsData(path="/data/wmucha/datasets",
                             imgs_path="h2o_ego/h2o_ego",
                             batch_size=1,
                             img_size=[224, 224],
                             norm_mean=[0.485, 0.456, 0.406],
                             norm_std=[0.229, 0.224, 0.225],
                             use_depth=False,
                             depth_img_type='est_low',
                             segment_list=[0, 90],
                             segment_list_val=[120],
                             )

    dataset = H2O_Dataset_hand_train_3D(spock_cfg, 'test')

    for i in tqdm(range(len(dataset))):

        batch = dataset[i]
        pth = batch['img_path']
        pth_depth = get_pth_for_h2o_depth(pth)

        # if file pth_depth exists, skip
        if os.path.exists(pth_depth):
            print('exist')
            continue

        img = np.array(Image.open(pth))
        inputs = feature_extractor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model_rgb2depth(**inputs)
        predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(384, 384),
            mode="bicubic",
            align_corners=False,
        )

        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        depth.save(pth_depth)


if __name__ == '__main__':
    main()
