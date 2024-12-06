# test the model for different-time data on event-image pair
from typing import Dict
import torch
import numpy as np
import cv2 as cv
import logging
import json
import os
import time
from rich import print
from datasets.MVSEC import MVSECDataset_RPE_VAL
from datasets.MVSEC import *
from datasets.EC import ECDataset_VAL
from datasets.visualize import draw_events_color_image
from core.modules.EIM import *

from omegaconf import OmegaConf
from utils.common import Padder
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.metrics.keypoints_metrics import Repeatability
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


def padding_data(data: Dict, padder: Padder) -> Dict:
    (
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    ) = padder.pad(
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    )
    return data


def unpad_data(data: Dict, padder: Padder) -> Dict:
    (
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    ) = padder.unpad(
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    )
    return data


def unpad_feats(feats: Dict, padder: Padder) -> Dict:
    feats["score"], feats["nms"], feats["normalized_descriptors"] = padder.unpad(
        feats["score"], feats["nms"], feats["normalized_descriptors"]
    )
    return feats


def draw_keypoints(image, keypoints):
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        cv.circle(image, (int(x), int(y)), 1, (0, 255, 0), 1)

    return image


def draw_matched_kpts(image0, image1, matched_kpts0, matched_kpts1, c='g', stacked_image=None):
    if stacked_image is None:
        stacked_image = np.hstack((image0, image1))
    for i in range(len(matched_kpts0)):
        x0, y0 = matched_kpts0[i]
        x1, y1 = matched_kpts1[i]
        if c == 'g':
            cv.line(
                stacked_image,
                (int(x0), int(y0)),
                (int(x1) + image0.shape[1], int(y1)),
                (0, 255, 0),
                1,
            )
        else:
            cv.line(
                stacked_image,
                (int(x0), int(y0)),
                (int(x1) + image0.shape[1], int(y1)),
                (0, 0, 255),
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
        dataset_cfg = OmegaConf.load("configs/dataset/test_mvsec.yaml")
        # indices path list
        indices_list = [
            "./indoor_flying4_final_indices.txt",
            "./outdoor_day1_final_indices.txt",
        ]
        dataset = MVSECDataset_RPE_VAL(dataset_cfg, indices_list)
    elif dataset_name == 'EC':
        # EC
        dataset_cfg = OmegaConf.load("configs/dataset/test_ec.yaml")
        dataset = ECDataset_VAL(dataset_cfg)
    else:
        raise ValueError("Invalid dataset name")

    # model: EIM
    model_name = 'sp-lg'
    model_cfg = OmegaConf.load("configs/model/SP_LG.yaml")
    model = EIM(model_cfg, "cuda")
    ckpt_path = "ckpts/MVSEC_EI_SP_LG_ft_Stage2.pth"
    model.load_state_dict(torch.load(ckpt_path), strict=False)
    model.eval()

    # metrics
    RPE = RelativePoseEstimation("RPE", pose_thresh=[5, 10, 20])

    # test the model in different-time data
    rpe_dict = {}
    pbar = tqdm(dataset)
    
    R_err_list = []
    t_err_list = []
    pose_err_list = []
    ransac_inlier_ratio_list = []
    matches_list = []
    
    count=0
    for data_pair in pbar:
        # if count != 16:
        #     count += 1
        #     continue
        
        data0, data1, T_0to1, T_1to0 = data_pair

        # load input into cuda
        events = data0["events_rep"].unsqueeze(0).cuda()
        image = data1["image"].unsqueeze(0).cuda()
        events_mask = (data0["events_image"].unsqueeze(0).cuda()) > 0

        # inference
        events_feats, image_feats, matches = model(events, image, events_mask)

        # get detected keypoints and matched keypoints
        e_positions = events_feats["sparse_positions"]
        i_positions = image_feats["sparse_positions"]
        e_positions = e_positions[0][None, :].cuda()
        i_positions = i_positions[0][None, :].cuda()

        # load camera and depth
        K0, K1 = data0['K'], data1['K']
        camera0, camera1 = Camera.from_calibration_matrix(K0), Camera.from_calibration_matrix(K1)
        camera0, camera1 = camera0.cuda(), camera1.cuda()
        # depth0 = data0['depth'].cuda()[None, :]
        # depth1 = data1['depth'].cuda()[None, :]

        # gt = gt_matches_from_pose_depth(
        #     kp0=e_positions,
        #     kp1=i_positions,
        #     camera0=camera0,
        #     camera1=camera1,
        #     depth0=depth0,
        #     depth1=depth1,
        #     T_0to1=Pose.from_4x4mat(T_0to1).cuda(),
        #     T_1to0=Pose.from_4x4mat(T_1to0).cuda(),
        #     ordering='yx'
        # )
        
        # true_matched_kpts0 = []
        # true_matched_kpts1 = []
        # false_matched_kpts0 = []
        # false_matched_kpts1 = []
        
        matched_kpts0 = matches['matched_kpts0'][0][..., :2]
        matched_kpts1 = matches['matched_kpts1'][0][..., :2]
        # print(f'matched_kpts0: {matched_kpts0.shape}')
        
        if model.event_extractor.extractor.ordering == "yx":
            matched_kpts0 = torch.flip(matched_kpts0, dims=[-1])
            matched_kpts1 = torch.flip(matched_kpts1, dims=[-1])
        
        # e_matches = matches['matches0'][0][0]
        # gt_matches0 = gt['matches0'][0]
        
        # for i in range(len(e_matches)):
        #     if e_matches[i] >=0 and gt_matches0[i] >= 0:
        #         if e_matches[i] == gt_matches0[i]:
        #             true_matched_kpts0.append(e_positions[0][i])
        #             true_matched_kpts1.append(i_positions[0][e_matches[i]])
        #         else:
        #             false_matched_kpts0.append(e_positions[0][i])
        #             false_matched_kpts1.append(i_positions[0][e_matches[i]])
                    
        # if len(true_matched_kpts0) == 0:
        #     R_err_list.append(100)
        #     t_err_list.append(100)
        #     pose_err_list.append(100)
        #     ransac_inlier_ratio_list.append(0)
        #     matches_list.append([0, 0, 0])
        #     count += 1
        #     continue
        
        # true_matched_kpts0 = torch.stack(true_matched_kpts0, dim=0).cpu().numpy()[:, :2]
        # true_matched_kpts1 = torch.stack(true_matched_kpts1, dim=0).cpu().numpy()[:, :2]
        # false_matched_kpts0 = torch.stack(false_matched_kpts0, dim=0).cpu().numpy()[:, :2]
        # false_matched_kpts1 = torch.stack(false_matched_kpts1, dim=0).cpu().numpy()[:, :2]
        # if model.event_extractor.extractor.ordering == "yx":
        #     true_matched_kpts0 = true_matched_kpts0[:, ::-1]
        #     true_matched_kpts1 = true_matched_kpts1[:, ::-1]
        #     false_matched_kpts0 = false_matched_kpts0[:, ::-1]
        #     false_matched_kpts1 = false_matched_kpts1[:, ::-1]

        rpe = RPE.update_one(
            matches["matched_kpts0"][0],
            matches["matched_kpts1"][0],
            data0["K"],
            data1["K"],
            T_0to1,
        )
        logging.info(rpe)
        for k, v in rpe.items():
            if k in rpe_dict.keys():
                rpe_dict[k].append(v)
            else:
                rpe_dict[k] = [v]
        
        print(f'R_error: {rpe["RPE_R_errs"]:.4f}, t_error: {rpe["RPE_t_errs"]:.4f}, pose_error: {rpe["RPE_pose_errs"]:.4f}, inlier_ratio: {rpe["RPE_inliers"]:.4f}')
        # print(f'matches: {len(true_matched_kpts0)}/{len(true_matched_kpts0)+len(false_matched_kpts0)} ({len(true_matched_kpts0)/(len(true_matched_kpts0)+len(false_matched_kpts0)):.4f})')
        R_err_list.append(rpe["RPE_R_errs"])
        t_err_list.append(rpe["RPE_t_errs"])
        pose_err_list.append(rpe["RPE_pose_errs"])
        ransac_inlier_ratio_list.append(rpe["RPE_inliers"])
        # matches_list.append([len(true_matched_kpts0), len(true_matched_kpts0)+len(false_matched_kpts0), len(true_matched_kpts0)/(len(true_matched_kpts0)+len(false_matched_kpts0))])

        # # convert image
        # events_image = data0["events_image"].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        # events_image = cv.cvtColor(events_image, cv.COLOR_GRAY2BGR)
        # image = data1["image"].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        # image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # events_mask_image = (events_image > 0).astype(np.uint8) * 128
        
        # # print(f'events num: {len(data0["events_raw"]["t"])}')
        # events_color_image = draw_events_color_image(data0['events_raw'], (image.shape[1], image.shape[0]))

        # # matched_image = draw_matched_kpts(
        # #     events_color_image.copy(), image.copy(), false_matched_kpts0, false_matched_kpts1, c='r'
        # # )
        # # matched_image = draw_matched_kpts(
        # #     events_color_image.copy(), image.copy(), true_matched_kpts0, true_matched_kpts1,  stacked_image=matched_image
        # # )
        
        # matched_image = draw_matched_kpts(
        #     events_color_image.copy(), image.copy(), matched_kpts0, matched_kpts1
        # )

        # plt.figure(1)
        # plt.subplot(1, 3, 1)
        # plt.imshow(kp_e_img)
        # plt.subplot(1, 3, 2)
        # plt.imshow(kp_i_img)
        # plt.subplot(1, 3, 3)
        # plt.imshow(events_mask_image)

        # plt.figure()
        # plt.imshow(matched_image)

        # plt.show()
        
        # if not os.path.exists(f'outputs/{dataset_name}/{model_name}/RPE/EI'):
        #     os.makedirs(f'outputs/{dataset_name}/{model_name}/RPE/EI')
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/RPE/EI/{count}_events_image.png', events_color_image)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/RPE/EI/{count}_image.png', image)
        # cv.imwrite(f'outputs/{dataset_name}/{model_name}/RPE/EI/{count}_matched_image.png', matched_image)
        count += 1

        # break
        
    

    # calculate auc
    auc = RPE.compute_all_auc()

    for k in rpe_dict.keys():
        v = np.array(rpe_dict[k])
        v = v[np.isfinite(v)]
        rpe_dict[k] = np.mean(v)

    for k in RPE.pose_thresh:
        rpe_dict[f"{RPE.metric_name}@{k}_auc"] = auc[f"{k}"]

    print("-------------")
    print("ALL:")
    print(rpe_dict)
    logging.info(rpe_dict)
    
    # R_err_list = np.array(R_err_list)
    # t_err_list = np.array(t_err_list)
    # pose_err_list = np.array(pose_err_list)
    # ransac_inlier_ratio_list = np.array(ransac_inlier_ratio_list)
    # matches_list = np.array(matches_list)
    # np.savetxt(f'outputs/{dataset_name}/{model_name}/RPE/EI/R_err_list.txt', R_err_list, fmt='%.4f')
    # np.savetxt(f'outputs/{dataset_name}/{model_name}/RPE/EI/t_err_list.txt', t_err_list, fmt='%.4f')
    # np.savetxt(f'outputs/{dataset_name}/{model_name}/RPE/EI/pose_err_list.txt', pose_err_list, fmt='%.4f')
    # np.savetxt(f'outputs/{dataset_name}/{model_name}/RPE/EI/ransac_inlier_ratio_list.txt', ransac_inlier_ratio_list, fmt='%.4f')
    # np.savetxt('outputs/RPE/EI_matches_list.txt', matches_list, fmt='%.4f')


if __name__ == "__main__":
    setup(42, True, False, 8)
    main()
