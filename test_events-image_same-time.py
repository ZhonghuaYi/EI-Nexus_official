# test the model for same-time data on events-image pair for the EDM model
from typing import Dict
import torch
import numpy as np
import cv2 as cv
import logging
import json
import os
from rich import print
from datasets.MVSEC import MVSECDataset_RPE_VAL
from datasets.EC import ECDataset_VAL
from datasets.visualize import draw_events_color_image
from core.modules.EIM import EIM

from omegaconf import OmegaConf
from utils.common import Padder
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.metrics.keypoints_metrics import Repeatability, ValidDescriptorsDistance
from core.metrics.matching_metrics import (
    MeanMatchingAccuracy,
    MatchingRatio,
    HomographyEstimation,
    RelativePoseEstimation,
    compute_auc,
)

from core.geometry.gt_generation import gt_matches_from_pose_depth
from core.geometry.wrappers import Camera, Pose
from core.modules.utils.detector_util import (
    logits_to_prob,
    depth_to_space,
    prob_map_to_points_map,
    prob_map_to_positions_with_prob,
    get_dense_positions,
)
from core.modules.utils.descriptor_util import (
    normalize_descriptors,
    get_dense_descriptors,
    sparsify_full_resolution_descriptors,
    sparsify_low_resolution_descriptors,
    upsample_descriptors,
)
from utils.common import setup


def draw_keypoints(image, keypoints, c='g'):
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        if c == 'g':
            cv.circle(image, (int(x), int(y)), 1, (0, 255, 0, 255), -1)
        else:
            cv.circle(image, (int(x), int(y)), 1, (0, 0, 255, 255), -1)

    return image


def draw_matched_kpts(image0, image1, matched_kpts0, matched_kpts1):
    stacked_image = np.hstack((image0, image1))
    for i in range(len(matched_kpts0)):
        x0, y0 = matched_kpts0[i]
        x1, y1 = matched_kpts1[i]
        cv.circle(stacked_image, (int(x0), int(y0)), 1, (0, 255, 0), 1)
        cv.circle(
            stacked_image, (int(x1) + image0.shape[1], int(y1)), 1, (0, 255, 0), 1
        )
        cv.line(
            stacked_image,
            (int(x0), int(y0)),
            (int(x1) + image0.shape[1], int(y1)),
            (0, 255, 0),
            1,
        )

    return stacked_image


@torch.no_grad()
def main():
    # log
    logging.basicConfig(
        filename=f"{__name__}.log",
        filemode="w",
        format="%(message)s",
        level=logging.INFO,
    )
    
    dataset_name = 'MVSEC'
    if dataset_name == 'MVSEC':
        ## MVSEC
        # dataset
        dataset_cfg = OmegaConf.load("configs/dataset/test_mvsec.yaml")
        # indices path list
        indices_list = [
            "./indoor_flying4_final_indices.txt",
            "./outdoor_day1_final_indices.txt",
        ]
        dataset = MVSECDataset_RPE_VAL(dataset_cfg, indices_list)
    elif dataset_name == 'EC':
        ## EC
        dataset_cfg = OmegaConf.load("configs/dataset/test_ec.yaml")
        dataset = ECDataset_VAL(dataset_cfg)
    else:
        raise ValueError("Invalid dataset name")

    # model: EIM
    model_name = 'sp-mnn'
    model_cfg = OmegaConf.load("configs/model/SP_MNN.yaml")
    model = EIM(model_cfg, "cuda")
    ckpt_path = "./ckpts/MVSEC_EI_SP_MNN_Stage1.pth"
    model.load_state_dict(torch.load(ckpt_path), strict=False)
    model.eval()

    # metrics
    # R_1 = Repeatability("repeatability@1", distance_thresh=1)
    # R_3 = Repeatability("repeatability@3", distance_thresh=3)
    VDD = ValidDescriptorsDistance("VDD", distance_thresh_list=[1, 3])
    MMA_1 = MeanMatchingAccuracy("MMA@1", threshold=1)
    MMA_3 = MeanMatchingAccuracy("MMA@3", threshold=3)
    MR = MatchingRatio("MR")
    HE = HomographyEstimation("HE", correctness_thresh=[3, 5, 10])
    RPE = RelativePoseEstimation("RPE", pose_thresh=[5, 10, 20])

    # test the model in different-time data
    result_dict = {}
    events_num_list = []
    pbar = tqdm(dataset)
    i=0
    for data_pair in pbar:
        data0, _, _, _ = data_pair

        # load input into cuda
        events_rep = data0["events_rep"].unsqueeze(0).cuda()
        image1 = data0["image"].unsqueeze(0).cuda()
        # events mask
        events_mask = (data0["events_image"].unsqueeze(0).cuda()) > 0
        # depth_mask = (data0["depth_mask"] > 0).unsqueeze(0).cuda()
        
        # print(f'events num: {len(data0["events_raw"]["t"])}')

        # inference
        events_feats, image_feats1, matches = model(events_rep, image1, events_mask)

        # get detected keypoints and matched keypoints
        i0_positions = events_feats["sparse_positions"]
        i1_positions = image_feats1["sparse_positions"]
        i0_matched_positions = matches["matched_kpts0"]
        i1_matched_positions = matches["matched_kpts1"]

        b = events_rep.shape[0]
        true_homography = torch.eye(3).repeat(b, 1, 1).cuda()
        # r_1 = R_1.update_one(
        #     events_feats["sparse_positions"][0],
        #     image_feats1["sparse_positions"][0],
        #     events_rep.shape[-2:],
        #     image1.shape[-2:],
        #     true_homography[0],
        # )
        # r_3 = R_3.update_one(
        #     events_feats["sparse_positions"][0],
        #     image_feats1["sparse_positions"][0],
        #     events_rep.shape[-2:],
        #     image1.shape[-2:],
        #     true_homography[0],
        # )
        vdd = VDD.update_one(
            events_feats["sparse_positions"][0],
            image_feats1["sparse_positions"][0],
            events_feats["sparse_descriptors"][0],
            image_feats1["sparse_descriptors"][0],
            image1.shape[-2:],
            image1.shape[-2:],
            true_homography[0],
        )

        mma_1 = MMA_1.update_one(
            matches["matched_kpts0"][0], matches["matched_kpts1"][0], true_homography[0]
        )
        mma_3 = MMA_3.update_one(
            matches["matched_kpts0"][0], matches["matched_kpts1"][0], true_homography[0]
        )
        mr = MR.update_one(
            matches["matched_kpts0"][0],
            matches["matched_kpts1"][0],
            events_feats["sparse_positions"][0],
            image_feats1["sparse_positions"][0],
        )
        he = HE.update_one(
            events_feats["image_size"][0],
            matches["matched_kpts0"][0],
            matches["matched_kpts1"][0],
            true_homography[0],
        )
        result = {
            # **r_1,
            # **r_3,
            **vdd,
            **mma_1,
            **mma_3,
            **mr,
            **he,
            # **rpe,
        }
        logging.info(result)
        for k, v in result.items():
            if k in result_dict.keys():
                result_dict[k].append(v)
            else:
                result_dict[k] = [v]

        # convert image
        image0 = data0['events_image'].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        image0 = cv.cvtColor(image0, cv.COLOR_GRAY2BGR)
        image1 = data0['image'].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        image1 = cv.cvtColor(image1, cv.COLOR_GRAY2BGR)
        
        events_mask_img = data0['events_image'][0].cpu().numpy().astype(np.uint8)
        events_mask_img = (events_mask_img * 255).astype(np.uint8) // 2
        events_mask_img = cv.cvtColor(events_mask_img, cv.COLOR_GRAY2BGR)
        
        # print(f'events num: {len(data0["events_raw"]["t"])}')
        events_num = len(data0['events_raw']['t'])
        events_num_list.append(events_num)
        events_color_image = draw_events_color_image(data0['events_raw'], (image0.shape[1], image0.shape[0]))

        # convert keypoints
        i0_positions = i0_positions[0][:, :2].cpu().numpy()
        i1_positions = i1_positions[0][:, :2].cpu().numpy()
        i0_matched_positions = i0_matched_positions[0][:, :2].cpu().numpy()
        i1_matched_positions = i1_matched_positions[0][:, :2].cpu().numpy()
        if model.image_extractor.extractor.ordering == 'yx':
            i0_positions = i0_positions[:, ::-1]
            i1_positions = i1_positions[:, ::-1]
            i0_matched_positions = i0_matched_positions[:, ::-1]
            i1_matched_positions = i1_matched_positions[:, ::-1]

        # # draw keypoints into image
        # repeat_index = vdd['valid_points1']
        # i0_positions_repeat = i0_positions[repeat_index]
        # i0_positions_no_repeat = i0_positions[~repeat_index]
        
        kp_i0_img = draw_keypoints(events_color_image.copy(), i0_positions)
        # kp_i0_img = draw_keypoints(kp_i0_img, i0_positions_no_repeat, c='r')
        kp_i1_img = draw_keypoints(image1.copy(), i1_positions)
        matched_image = draw_matched_kpts(image0.copy(), image1.copy(), i0_matched_positions, i1_matched_positions)

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(kp_i0_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(kp_i1_img)
        
        # if not os.path.exists(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI'):
        #     os.makedirs(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI')
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI/{i}_matched_image.png', matched_image)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI/{i}_events.png', events_color_image)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI/{i}_events_mask.png', events_mask_img)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI/{i}_image.png', image1)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI/{i}_event_kpt.png', kp_i0_img)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/KeypointsExtraction/EI/{i}_image_kpt.png', kp_i1_img)
        i+=1
        # plt.figure()
        # plt.imshow(matched_image)

        # plt.show()

    # # calculate auc
    he_auc = HE.compute_all_auc()

    for k in result_dict.keys():
        v = np.array(result_dict[k])
        v = v[np.isfinite(v)]
        result_dict[k] = np.mean(v)

    for k in HE.correctness_thresh:
        result_dict[f"{HE.metric_name}@{k}_auc"] = he_auc[f"{k}"]

    print("-------------")
    print("ALL:")
    print(result_dict)
    print(f'events num: {np.mean(events_num_list)}')
    logging.info(result_dict)


if __name__ == "__main__":
    setup(42, False, False, 8)
    main()
