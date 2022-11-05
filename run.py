from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import cv2
import argparse

if os.getenv('SYSTEM') == 'spaces':
    import mim

    mim.uninstall('mmcv-full', confirm_yes=True)
    mim.install('mmcv-full==1.5.0', is_yes=True)

    subprocess.run('pip uninstall -y opencv-python'.split())
    subprocess.run('pip uninstall -y opencv-python-headless'.split())
    subprocess.run('pip install opencv-python-headless==4.5.5.64'.split())

import huggingface_hub
import numpy as np
import torch
import torch.nn as nn
from time import time
app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'ViTPose/'
sys.path.insert(0, app_dir.as_posix())

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)

HF_TOKEN = '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' # os.environ['HF_TOKEN']


class DetModel:
    MODEL_DICT = {
        'YOLOX-tiny': {
            'config':
            '/home/adithyas/mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        },
        'YOLOX-s': {
            'config':
            '/home/adithyas/mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        },
        'YOLOX-l': {
            'config':
            '/home/adithyas/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        },
        'YOLOX-x': {
            'config':
            '/home/adithyas/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        },
        'SWIN-T':{
          'config':
            '/home/adithyas/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'    
        },
        'SWIN-S':{
          'config':
            '/home/adithyas/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'    
        },
        # 'SWIN-B':{
        #   'config':
        #     '/home/adithyas/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py',
        #     'model':
        #     '/home/adithyas/ViTPose/weights/swin_base.pth'    
        # },
        # 'ViT-B':{
        #   'config':
        #     '/home/adithyas/ViTDet/configs/ViTDet/ViTDet-ViT-Base-100e.py',
        #     'model':
        #     '/home/adithyas/ViTPose/weights/ViT-Base-GPU.pth'    
        # },
        # 'ViTAE-B':{
        #   'config':
        #     '/home/adithyas/ViTDet/configs/ViTDet/ViTDet-ViTAE-Base-100e.py',
        #     'model':
        #     '/home/adithyas/ViTPose/weights/ViTAE-Base-GPU.pth'    
        # },
    }

    def __init__(self, device: str | torch.device):
        device = torch.device("cuda")
        self.device = torch.device(device)
        # self._load_all_models_once()
        self.model_name = 'SWIN-S'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        return init_detector(dic['config'], dic['model'], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
                
        if "YOLO" in self.model_name:
            person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79
        elif "SWIN" in self.model_name:
            person_det = detection_results[0]
            
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(image,
                                     person_det,
                                     score_thr=score_threshold,
                                     bbox_color=None,
                                     text_color=(200, 200, 200),
                                     mask_color=None)
        return vis[:, :, ::-1] # BGR -> RGB


class AppDetModel(DetModel):
    def run(self, model_name: str, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        self.set_model(model_name)
        return self.detect_and_visualize(image, score_threshold)


class PoseModel:
    #  MODEL_DICT = {
    #     'ViTPose-B (single-task train)': {
    #         'config':
    #         '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
    #         'model': 'models/vitpose-b.pth',
    #     },
    #     'ViTPose-L (single-task train)': {
    #         'config':
    #         '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
    #         'model': 'models/vitpose-l.pth',
    #     },
    #     'ViTPose-B (multi-task train, COCO)': {
    #         'config':
    #         '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
    #         'model': 'models/vitpose-b-multi-coco.pth',
    #     },
    #     'ViTPose-L (multi-task train, COCO)': {
    #         'config':
    #         '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
    #         'model': 'models/vitpose-l-multi-coco.pth',
    #     },
    #     'ViTPose-H (multi-task train, COCO)': {
    #         'config':
    #         '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
    #         'model': 'models/vitpose-h-multi-coco.pth',
    #     },
    # }
    
    MODEL_DICT = {
        'ViTPose-B (single-task train)': {
            'config':
            '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': '/home/adithyas/ViTPose/models/vitpose-b.pth',
        },
        'ViTPose-L (single-task train)': {
            'config':
            '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': '/home/adithyas/ViTPose/models/vitpose-l.pth',
        },
        'ViTPose-B (multi-task train, COCO)': {
            'config':
            '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': '/home/adithyas/ViTPose/models/vitpose-b-multi-coco.pth',
        },
        'ViTPose-L (multi-task train, COCO)': {
            'config':
            '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': '/home/adithyas/ViTPose/models/vitpose-l-multi-coco.pth',
        },
        'ViTPose-H (multi-task train, COCO)': {
            'config':
            '/home/adithyas/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
            'model': '/home/adithyas/ViTPose/models/vitpose-h-multi-coco.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        device = torch.device("cuda")
        self.device = torch.device(device)
        self.model_name = 'ViTPose-H (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model'] # huggingface_hub.hf_hub_download('hysts/ViTPose', dic['model'], use_auth_token=HF_TOKEN)
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold, vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  
        # BGR -> RGB


class AppPoseModel(PoseModel):
    def run(
        self, model_name: str, image: np.ndarray,
        det_results: list[np.ndarray], box_score_threshold: float,
        kpt_score_threshold: float, vis_dot_radius: int,
        vis_line_thickness: int
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        self.set_model(model_name)
        return self.predict_pose_and_visualize(image, det_results,
                                               box_score_threshold,
                                               kpt_score_threshold,
                                               vis_dot_radius,
                                               vis_line_thickness)

def main(args):
    det_model = AppDetModel("cuda")
    pose_model = AppPoseModel("cuda")
    
    video = cv2.VideoCapture(args.video_file)
    if (video.isOpened() == False): 
        print("Error reading video file")
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)
    
    # 'XVID' 'MP42' 'DIVX'->(Windows) - AVI
    # 'MP4V' 'MJPG' 'VIDX' - MP4
    result = cv2.VideoWriter(args.save_file, 
                            cv2.VideoWriter_fourcc(*'{}'.format(args.codec)),
                            10, size)
        
    while(True):
        ret, frame = video.read()
        if ret == True: 
            start = time()
            # from pdb import set_trace; set_trace()
            det_out, det_vis = det_model.run(args.detmodel_name, frame, args.box_score_threshold)
            pose_out, pose_vis = pose_model.run(args.posemodel_name, frame, det_out, args.box_score_threshold, \
                        args.kpt_score_threshold, args.vis_dot_radius, args.vis_line_thickness)
            end = time() - start
            print("Time for frame: {}".format(end))
            result.write(pose_vis)
    
        # Break the loop
        else:
            print("Video complete")
            break
  
    video.release()
    result.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, default="/home/adithyas/ViTPose/input/10.mp4")
    parser.add_argument("--save_file", type=str, default="/home/adithyas/ViTPose/output/10_out_H.mp4")
    parser.add_argument("--codec", type=str, default="MJPG")
    parser.add_argument("--detmodel_name", type=str, default="SWIN-S")
    parser.add_argument("--box_score_threshold", type=float, default=0.3)
    parser.add_argument("--posemodel_name", type=str, default="ViTPose-H (multi-task train, COCO)")
    parser.add_argument("--kpt_score_threshold", type=float, default=0.2)
    parser.add_argument("--vis_dot_radius", type=int, default=4)
    parser.add_argument("--vis_line_thickness", type=int, default=2)
    args = parser.parse_args()
    main(args)
    
    
