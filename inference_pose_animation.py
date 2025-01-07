import os
import cv2
import random
import time
from datetime import datetime
import torch.nn.functional as F

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
from torchvision import transforms

import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    AutoencoderKL
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from src.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

from src.pipelines.pipeline_svd_pose_animation import StableVideoDiffusionPipeline
from src.models.pose_guider import PoseGuider

from src.dwpose import DWposeDetector, draw_pose

from src.utils.utils import (
    get_fps, 
    read_frames, 
    save_videos_grid,
)



class AnimateController:
    def __init__(
        self,
        config_path="./configs/inference/pose_animation.yaml",
        weight_dtype=torch.float16,
    ):
        # Read pretrained weights path from config
        self.cfg = OmegaConf.load(config_path)
        self.pipeline = None
        self.weight_dtype = weight_dtype
        self.dwpose_processor = DWposeDetector()
   
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="vae")
        sd_vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_sd_model_name_or_path, 
            subfolder="vae")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="image_encoder")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="unet")
        
        reference_net = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_sd_model_name_or_path,
            subfolder="unet",
        ).to(device="cuda")

        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, 
            block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")

        # load pretrained weights
        pose_guider.load_state_dict(
            torch.load(self.cfg.pose_guider_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        # load pretrained weights
        reference_net.load_state_dict(
            torch.load(self.cfg.reference_net_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        unet.load_state_dict(
            torch.load(self.cfg.unet_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        self.checkpoint_step = self.cfg.unet_checkpoint_path.split("/")[-1].split(".")[0].split("-")[1]

        image_encoder.to(self.weight_dtype)
        vae.to(self.weight_dtype)
        sd_vae.to(self.weight_dtype)
        pose_guider.to(self.weight_dtype)
        reference_net.to(self.weight_dtype)
        unet.to(self.weight_dtype)

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            unet=unet,
            reference_net=reference_net,
            image_encoder=image_encoder,
            vae=vae,
            sd_vae=sd_vae,
            pose_guider=pose_guider,
        )
        pipe = pipe.to("cuda", dtype=self.weight_dtype)
        
        self.pipeline = pipe
    

    def prepare_ref_pose(
        self, 
        ref_img_path, 
        save_path=None,
    ):
        image_bgr = cv2.imread(ref_img_path)
        h, w, c = image_bgr.shape
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            detected_pose, detected_pose_score = self.dwpose_processor.det_and_draw_kpts24(image)
        return detected_pose


    def animate(
        self,
        ref_img_path, 
        dwpose_video_path,
        height=968,
        width=648,
        num_inference_steps=25,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        overlap=4,
        fps=7,
        noise_aug_strength=0.02,
        frames_per_batch=16,
        motion_bucket_id=20,
        decode_chunk_size=8,
        seed=123,
    ):
        generator = torch.manual_seed(seed) 
        
        checkpoint_step = self.checkpoint_step
        print("pipeline done")
        tgt_h = height
        tgt_w = width

        ref_name = Path(ref_img_path).stem
        pose_name = Path(dwpose_video_path).stem.replace("_kps", "")

        ref_image_pil = Image.open(ref_img_path).convert("RGB")
        ref_w_ori, ref_h_ori = ref_image_pil.size

        save_dir = f"./output/{ref_name}_{pose_name}_{height}_{width}_{checkpoint_step}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # get ref_pose
        ref_pose_np = self.prepare_ref_pose(ref_img_path)
        cv2.imwrite(f"./{save_dir}/{ref_name}_pose_ori.png", ref_pose_np)
        
        def get_bounding_box(ref_pose_np):
            ref_gray_pose = np.mean(ref_pose_np, axis=2)
            ref_mask = (ref_gray_pose > 0.05).astype(np.float32)  
            y_coords, x_coords = np.nonzero(ref_mask)
            left, right = x_coords.min(), x_coords.max()
            top, bottom = y_coords.min(), y_coords.max()
            return left, right, top, bottom
        
        def extend_rectangle(
            left, 
            right, 
            top, 
            bottom, 
            ref_w, 
            ref_h, 
            scale_ratio=0.2,
        ):
            human_h = bottom - top
            human_w = right - left

            # left = int(left - scale_ratio * human_w)
            # right = int(right + scale_ratio * human_w)

            top = int(top - scale_ratio * human_h)
            bottom = int(bottom + scale_ratio * human_h)

            left = max(0, left)
            right = min(ref_w, right)
            top = max(0, top)
            bottom = min(ref_h, bottom)
            return left, right, top, bottom
        
        left, right, top, bottom = get_bounding_box(ref_pose_np)

        left, right, top, bottom = extend_rectangle(
           left, right, top, bottom, ref_w_ori, ref_h_ori
        )
        
        # left right top bottom width
        left_width = max(0, left)
        right_width = max(ref_w_ori - right, 0)
        
        top_width = max(0, top)
        bottom_width = max(0, ref_h_ori - bottom)

        if (ref_h_ori / ref_w_ori) < (tgt_h / tgt_w):
            crop_h = ref_h_ori
            crop_w = int(ref_h_ori / tgt_h * tgt_w)

            
            crop_left = int((ref_w_ori - crop_w) * (left_width / (left_width + right_width)))
            crop_right = int((ref_w_ori - crop_w) * (right_width / (left_width + right_width)))

            crop_left = crop_left
            crop_right = ref_w_ori - crop_right
            crop_top = 0
            crop_bottom = ref_h_ori
            # crop image and pose
            ref_image_pil = ref_image_pil.crop(
                (crop_left, crop_top, crop_right, crop_bottom))
            ref_pose_np = ref_pose_np[crop_top:crop_bottom, crop_left:crop_right, :]

        else:
            crop_h = int(ref_w_ori / tgt_w * tgt_h)
            crop_w = ref_w_ori
           
            crop_top = int((ref_h_ori - crop_h) * (top_width / (top_width + bottom_width + 1e-8)))
            crop_bottom = int((ref_h_ori - crop_h) * (bottom_width / (top_width + bottom_width + 1e-8)))

            crop_top = crop_top
            crop_bottom = ref_h_ori - crop_bottom
            crop_left = 0
            crop_right = ref_w_ori

            # crop image and pose
            ref_image_pil = ref_image_pil.crop(
                (crop_left, crop_top, crop_right, crop_bottom))
            ref_pose_np = ref_pose_np[crop_top:crop_bottom, crop_left:crop_right, :]
       
        print("crop image and pose")
        ref_image_pil.save(f"./{save_dir}/{ref_name}_img_crop.png")
        cv2.imwrite(f"./{save_dir}/{ref_name}_pose_crop.png", ref_pose_np)

        
        print(f"infer width {width} and height {height}")

        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(dwpose_video_path) # pose
        src_fps = get_fps(dwpose_video_path)
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
    
        def get_bbox(frame):
            gray = np.mean(frame, axis=2)
            val = np.nonzero(gray > 25.5)
            top, left = np.min(val, axis=1)
            bottom, right = np.max(val, axis=1)
            # use face top as body top, face l & r
            val_face = np.nonzero(gray > 200) # body brightest on gray is around 178
            top, left = np.min(val_face, axis=1)
            _, right = np.max(val_face, axis=1)

            return (top, left, bottom, right)  
     
        ref_pose_np = cv2.resize(ref_pose_np, (width, height))
        ref_pose_bbox = get_bbox(ref_pose_np)

        ref_pose_h = ref_pose_bbox[2] - ref_pose_bbox[0]
        ref_pose_cx = (ref_pose_bbox[1] + ref_pose_bbox[3]) / 2
        ref_pose_cy = (ref_pose_bbox[0] + ref_pose_bbox[2]) / 2

        if True:
            print("Use avg bbox")
            pose_bboxs = [get_bbox(np.array(pose_image)) for pose_image in pose_images]
            pose_bbox = np.mean(np.array(pose_bboxs), axis=0)

        else:
            print("Use first bbox")
            pose_image_np = np.array(pose_images[0])
            pose_bbox = get_bbox(pose_image_np)

        pose_h = pose_bbox[2] - pose_bbox[0]
        pose_cx = (pose_bbox[1] + pose_bbox[3]) / 2
        pose_cy = (pose_bbox[0] + pose_bbox[2]) / 2

        # pose to ref_pose
        scale = ref_pose_h / pose_h
        off_x = ref_pose_cx - scale * pose_cx
        off_y = ref_pose_cy - scale * pose_cy

        trans = np.array([
            [scale, 0, off_x],
            [0, scale, off_y]])

        align_pose_images = []
        for pose_image_pil in pose_images:
            pose_image_np = np.array(pose_image_pil)
            pose_image_scale = cv2.warpAffine(pose_image_np, trans, (width, height))
            align_pose_images.append(Image.fromarray(pose_image_scale))

        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
            
        nframes = len(align_pose_images)
            
        for pose_image_pil in align_pose_images:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=nframes
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        start_time = time.time()
        with torch.cuda.amp.autocast(enabled=True):
            frames = self.pipeline(
                ref_image_pil,
                pose_list,
                height=height,
                width=width,
                num_frames=nframes,
                decode_chunk_size=decode_chunk_size,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                noise_aug_strength=noise_aug_strength,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                tile_overlap=overlap,
                num_inference_steps=num_inference_steps,
                tile_size=frames_per_batch,
            ).frames[0]
            print("svd pose2vid ellapsed: ", (time.time() - start_time) * 1000)

        video_np = np.stack([np.asarray(frame) / 255.0 for frame in frames])
        video = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0)
        
        video_shape = video.shape
        print('video shape:', video_shape)
        print('pose_tensor', pose_tensor.min(), pose_tensor.max())
        print('video', video.min(), video.max())
        
        video_result = F.interpolate(
            video, 
            size=(video_shape[2], height, width), 
            mode='trilinear', 
            align_corners=False).cpu()

        video = torch.cat([ref_image_tensor, pose_tensor, video_result], dim=0)
       
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        save_videos_grid(
            video, 
            f"{save_dir}/{date_str}_{time_str}_{num_inference_steps}_{frames_per_batch}_{overlap}_{motion_bucket_id}.mp4", 
            n_rows=3, 
            fps=src_fps)
        
        save_videos_grid(
            pose_tensor, 
            f"{save_dir}/pose.mp4", 
            n_rows=1, 
            fps=src_fps)


controller = AnimateController()


if __name__ == "__main__":
    ref_img_paths = [
        "test_data/ref_images/kehu001.jpg",
        "test_data/ref_images/kehu002.jpg",
        "test_data/ref_images/kehu003.jpg",
    ]

    dwpose_video_paths = [
        "test_data/pose_drive_videos/cloud006_dw24_15fps.mp4",
        "test_data/pose_drive_videos/cloud006_dw24_15fps.mp4",
        "test_data/pose_drive_videos/cloud006_dw24_15fps.mp4",
       
    ]

    for ref_img_path, dwpose_video_path in \
        zip(ref_img_paths, dwpose_video_paths):
        print(f"inference on {ref_img_path}, pose on {dwpose_video_path}")
        controller.animate(ref_img_path, dwpose_video_path)
