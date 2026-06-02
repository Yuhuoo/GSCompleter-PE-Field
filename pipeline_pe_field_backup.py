import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import cv2
import math
from PIL import Image
from contextlib import nullcontext
import torchvision.transforms as transforms
from jaxtyping import Float, UInt8
from typing import Union
from pathlib import Path
from einops import rearrange, repeat
from matplotlib.figure import Figure
from torch import Tensor
FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

# --- 1. 环境与 Import 设置 (严格还原) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    # 既然你有这些文件，这里必须直接 import
    from pvd_utils import random_sample_poses, world_point_to_obj, convert_opencv_to_nerf_c2w, convert_opencv_to_nerf_pointmap
    from diffusers import FluxKontextPipeline, FluxTransformer2DModel
    from moge.model import import_model_class_by_version
except ImportError as e:
    print(f"Import Error: {e}")
    print("请确保 'pvd_utils.py' 和 'moge' 文件夹在当前目录或 PYTHONPATH 中。")
    raise e

# --- 2. 原始 Helper 函数 (完全 Copy) ---

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688),
    (1568, 672),
]

def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()

def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)

def split_into_2x2_local_grids(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor of shape (H, W, 3) or (H//2, W//2, 3)
    Returns:
        Tensor of shape (H//2, W//2, 4*C) - 保持3维结构以支持递归调用
    """
    H, W, C = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Height and Width must be divisible by 2"

    # Step 1: reshape into 2x2 blocks
    x = x.view(H // 2, 2, W // 2, 2, C)  # → [H//2, 2, W//2, 2, 3]
    x = x.permute(0, 2, 1, 3, 4)         # → [H//2, W//2, 2, 2, 3]
    x = x.reshape(H // 2, W // 2, 4* C)  # → [H//2, W//2, 4*C]
    
    return x

def project_to_pixel(xyz_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, z = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=-1), z

def forward_splat(src_img: torch.Tensor, pix_coords: torch.Tensor, tgt_H: int, tgt_W: int, importance: torch.Tensor = None):
    B, C, H, W = src_img.shape
    device = src_img.device

    y_src, x_src = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    x_src = x_src.flatten()
    y_src = y_src.flatten()

    tgt_img = torch.zeros((B, C, tgt_H, tgt_W), device=device)
    weight_img = torch.zeros((B, 1, tgt_H, tgt_W), device=device)

    for b in range(B):
        tgt_x = pix_coords[b, :, 0]
        tgt_y = pix_coords[b, :, 1]
        tgt_x_int = tgt_x.round().long().clamp(0, tgt_W - 1)
        tgt_y_int = tgt_y.round().long().clamp(0, tgt_H - 1)

        if importance is not None:
            weights = importance[b, :, 0]
        else:
            weights = torch.ones_like(tgt_x)

        src_vals = src_img[b, :, y_src, x_src]

        for c in range(C):
            tgt_img[b, c].index_put_(
                (tgt_y_int, tgt_x_int),
                src_vals[c] * weights,
                accumulate=True
            )

        weight_img[b, 0].index_put_(
            (tgt_y_int, tgt_x_int),
            weights,
            accumulate=True
        )

    eps = 1e-6
    tgt_img = tgt_img / (weight_img + eps)
    return tgt_img

# --- 3. Pipeline Class ---

class PEFieldPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 路径配置
        self.moge_ckpt = os.path.join(BASE_DIR, "moge-2-vitl-normal", "model.pt")
        self.transformer_ckpt = os.path.join(BASE_DIR, "checkpoints")
        self.flux_path = os.path.join(BASE_DIR, "FLUX.1-Kontext-dev")
        
        self._init_models()

    def _init_models(self):
        print(f"Loading Moge from {self.moge_ckpt}...")
        self.moge_model = import_model_class_by_version("v2").from_pretrained(self.moge_ckpt).to(self.device).eval()
        
        print(f"Loading Flux Transformer from {self.transformer_ckpt}...")
        transformer = FluxTransformer2DModel.from_pretrained(self.transformer_ckpt, subfolder="transformer", torch_dtype=torch.bfloat16)
        
        print(f"Loading Flux Pipeline from {self.flux_path}...")
        self.pipe = FluxKontextPipeline.from_pretrained(self.flux_path, transformer=transformer, torch_dtype=torch.bfloat16)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, image_tensor: torch.Tensor, phi, theta, r=0, num_inference_steps=20):
        """
        Args:
            image_tensor: [1, 3, H, W], float32, [0, 1], device=cuda
        Returns:
            image_gen_tensor: [1, 3, H, W], float32, [0, 1], device=cuda (分辨率与输入严格一致)
            warped_image_tensor: [1, 3, H, W], float32, [0, 1], device=cuda
        """
        # 1. 记录原始信息，确保输入在 Device 上
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        
        # 记录原始维度 [1, 3, H, W]
        assert image_tensor.ndim == 4 and image_tensor.shape[0] == 1, f"Input shape must be [1, 3, H, W], got {image_tensor.shape}"
        orig_H, orig_W = image_tensor.shape[2], image_tensor.shape[3]
        
        # Moge 推理使用的输入 (squeeze掉batch维度以适配某些逻辑，或直接用)
        # 假设 moge.infer 接受 [C, H, W] 或 [1, C, H, W]。为了安全，处理成 [3, H, W] 进行推理前的准备
        img_input_moge = image_tensor.squeeze(0) # [3, H, W]
        C, H, W = img_input_moge.shape # 这里 H, W = orig_H, orig_W

        # 2. Moge 深度估计
        with torch.no_grad():
            moge_output = self.moge_model.infer(image_tensor, fov_x=None, resolution_level=9, num_tokens=None, use_fp16=False)
        
        depth = moge_output['depth'].cpu().numpy()
        intrinsics = moge_output['intrinsics'].cpu().numpy()

        # [FIX] 去掉 Batch 维度
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0] # [1, 3, 3] -> [3, 3]

        # [FIX] 深度图处理：可能返回 [1, H, W] 或 [H, W, 1]
        if depth.ndim == 3:
            if depth.shape[0] == 1: # [1, H, W]
                depth = depth.squeeze(0)
            elif depth.shape[2] == 1: # [H, W, 1]
                depth = depth.squeeze(2)

        depth[np.isinf(depth)] = np.max(depth[np.isfinite(depth)])

        # 3. 基础参数计算
        # 现在 intrinsics 是 (3, 3)，可以安全访问 [0,0] 和 [1,1]
        c2w_tensor = torch.eye(4, device=self.device)
        cx, cy = W / 2, H / 2
        fx = intrinsics[0, 0] * W
        fy = intrinsics[1, 1] * H

        # 4. 确定 Flux Context 的目标分辨率 (必须是 16 的倍数)
        aspect_ratio = W / H
        _, tgt_w, tgt_h = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
        
        H_news_W_news = [
            (tgt_h // 16, tgt_w // 16),
            (tgt_h // 8, tgt_w // 8),
            (tgt_h // 4, tgt_w // 4)
        ]

        # 5. 构建 Pointmap (2D -> 3D)
        x_range = np.arange(W)
        y_range = np.arange(H)
        xx, yy = np.meshgrid(x_range, y_range)

        x_cam = (xx - cx) * depth / fx
        y_cam = (yy - cy) * depth / fy
        pointmap = np.stack((x_cam, y_cam, depth), axis=-1)
        
        depth_avg = depth[H//2, W//2]
        radius = depth_avg * 1.0 
        
        pointmap_tensor = torch.from_numpy(pointmap).float().to(self.device)

        # 6. 几何变换 (pvd_utils)
        c2ws, pcd = world_point_to_obj(
            poses=c2w_tensor.unsqueeze(0), 
            points=pointmap_tensor.unsqueeze(0), 
            k=-1, 
            r=radius, 
            elevation=0., 
            device=self.device
        )

        frame_number = 1
        phi_list, theta_list, r_list = [phi, phi], [theta, theta], [r, r]
        
        new_c2ws = random_sample_poses(c2ws, phi_list, theta_list, r_list, frame_number, self.device)
        new_w2cs = torch.inverse(new_c2ws) 
        
        # 7. 3D 投影回 2D
        xyz_tgt_cam = (new_w2cs @ torch.cat([pcd.reshape(-1, 3), torch.ones(H*W, 1, device=self.device)], dim=1).T).T[:, :3]
        xyz_tgt_cam = xyz_tgt_cam.reshape(H, W, 3).unsqueeze(0)

        K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intr = torch.eye(4)
        intr[:3, :3] = torch.tensor(K_np)

        uv, z = project_to_pixel(xyz_tgt_cam.cpu().numpy(), intr.numpy())

        # 8. 深度归一化 & 坐标构建
        z_tensor = torch.from_numpy(z).float()
        z_min, z_max = z_tensor.min(), z_tensor.max()
        z_norm = (z_tensor - z_min) / (z_max - z_min + 1e-6) if (z_max - z_min) > 1e-6 else torch.zeros_like(z_tensor) + 0.01

        pix_coords = np.concatenate([z_norm.unsqueeze(-1).numpy(), uv], axis=-1) 
        pix_coords_tensor = torch.from_numpy(pix_coords).float() 
        pix_coords_tensor[..., 0] += 1.0 
        pix_coords_tensor = pix_coords_tensor.permute(0, 3, 1, 2) # [1, 3, H, W]

        # 9. Multi-scale Grid Loop (Conditioning)
        pix_coords_downs = []
        grid_level = 0
        
        for H_new, W_new in H_news_W_news:
            pix_coords_down = F.interpolate(pix_coords_tensor, size=(H_new, W_new), mode='bilinear', align_corners=False)
            pix_coords_down = pix_coords_down.squeeze(0).permute(1, 2, 0) # [H, W, 3]

            pix_coords_down[..., -2] = pix_coords_down[..., -2] / W * H_news_W_news[0][1]
            pix_coords_down[..., -1] = pix_coords_down[..., -1] / H * H_news_W_news[0][0]

            pix_coords_down = pix_coords_down[..., [0, 2, 1]] # Reorder

            if grid_level > 0:
                for _ in range(grid_level): 
                    pix_coords_down = split_into_2x2_local_grids(pix_coords_down)
            
            pix_coords_down = pix_coords_down.view(-1, pix_coords_down.shape[-1])
            grid_level += 1
            pix_coords_downs.append(pix_coords_down.to(self.device).to(torch.bfloat16))

        # 10. 生成 Warped Tensor [1, 3, orig_H, orig_W]
        # uv_flat 需要对应原始分辨率
        uv_flat = torch.from_numpy(uv).view(1, -1, 2).float().to(self.device)
        # forward_splat 输出 [1, 3, H, W]，值域 [0, 1] (因为输入 image_tensor 是 0-1)
        warped_image_tensor = forward_splat(image_tensor, uv_flat, orig_H, orig_W)

        # 11. Flux Inference
        # Resize 到 Flux 偏好的分辨率 (Target Res)
        image_resized = F.interpolate(image_tensor, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
        # Flux 需要 [-1, 1]
        masked_image = (image_resized - 0.5) / 0.5

        autocast_ctx = nullcontext()
        with autocast_ctx:
            # 返回 PIL List
            image_gen_pil = self.pipe(
                image=masked_image.to(dtype=torch.bfloat16), 
                height=tgt_h, 
                width=tgt_w,
                prompt="Ensure that the picture does not change in any way",
                input_img_ids=pix_coords_downs,
                use_multi_scale_position=True,
                num_inference_steps=num_inference_steps
            ).images[0]
            
            # --- 后处理：确保生成图像 Tensor 与输入形状完全一致 ---
            
            # 1. PIL -> Numpy -> Tensor [H_flux, W_flux, 3]
            gen_np = np.array(image_gen_pil)
            gen_tensor = torch.from_numpy(gen_np).float().to(self.device)
            
            # 2. [H, W, 3] -> [1, 3, H, W]
            gen_tensor = gen_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # 3. Normalize [0, 255] -> [0, 1]
            gen_tensor = gen_tensor / 255.0
            
            # 4. Resize 回原始分辨率 [1, 3, orig_H, orig_W]
            # 这里的 interpolate 会处理尺寸对齐
            if (tgt_h != orig_H) or (tgt_w != orig_W):
                image_gen_tensor = F.interpolate(gen_tensor, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            else:
                image_gen_tensor = gen_tensor
            
            # 5. 确保值域安全
            image_gen_tensor = image_gen_tensor.clamp(0, 1)

            return image_gen_tensor, warped_image_tensor

if __name__ == "__main__":
    # 使用示例
    pipeline = PEFieldPipeline()
    
    TEST_IMG_PATH = "outputs/12.15/0bfd-test-10/000004_original.png" 
    pil_img = Image.open(TEST_IMG_PATH).convert("RGB")
    img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 运行
    image_gen_tensor, warped_image_tensor = pipeline(img_tensor, phi=10, theta=5, r=0)

    save_image(img_tensor[0], "./debug/ori_image.png")
    save_image(image_gen_tensor[0], "./debug/warped_image.png")
    save_image(warped_image_tensor[0], "./debug/warped_pcd_image.png")