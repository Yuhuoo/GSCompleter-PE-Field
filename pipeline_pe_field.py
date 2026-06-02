import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from pvd_utils import random_sample_poses, world_point_to_obj, convert_opencv_to_nerf_c2w, convert_opencv_to_nerf_pointmap
    from diffusers import FluxKontextPipeline, FluxTransformer2DModel
    from moge.model import import_model_class_by_version
except ImportError as e:
    print(f"Import Error: {e}")
    raise e

class PEFieldPipeline:
    def __init__(self, device=None, H_orig=256, W_orig=448) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.moge_ckpt = os.path.join(BASE_DIR, "moge-2-vitl-normal", "model.pt")
        self.transformer_ckpt = os.path.join(BASE_DIR, "checkpoints")
        self.flux_path = os.path.join(BASE_DIR, "FLUX.1-Kontext-dev")
        self.H_orig = H_orig
        self.W_orig = W_orig
        
        self.preferred_resolutions = [
            (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
            (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
            (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
            (1504, 688), (1568, 672),
        ]
        
        self._init_models()

    def _init_models(self):
        print(f"Loading Moge from {self.moge_ckpt}...")
        self.moge = import_model_class_by_version("v2").from_pretrained(self.moge_ckpt).to(self.device).eval()
        
        print(f"Loading Flux Transformer from {self.transformer_ckpt}...")
        transformer = FluxTransformer2DModel.from_pretrained(self.transformer_ckpt, subfolder="transformer", torch_dtype=torch.bfloat16)
        
        print(f"Loading Flux Pipeline from {self.flux_path}...")
        self.pipe = FluxKontextPipeline.from_pretrained(self.flux_path, transformer=transformer, torch_dtype=torch.bfloat16)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        print("Pre-calculate mesh grid.")
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.H_orig, device=self.device, dtype=torch.float32),
            torch.arange(self.W_orig, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self._cached_grid = (y_grid, x_grid)
        
        print("Compling need cost more time.")
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="reduce-overhead")
        self._warm_up()
        print("Warm-up complete. System is ready.")

    def _warm_up(self):
        """使用伪数据触发 torch.compile 的实际编译过程"""
        print("Starting Warm-up inference (this triggers the actual compilation)...")
        with torch.inference_mode():
            # 1. 构造与 main 函数逻辑一致的输入 Tensor
            # 使用初始化时指定的 H_orig, W_orig
            img_tensor = torch.zeros((1, 3, self.H_orig, self.W_orig), device=self.device, dtype=torch.float32)
            
            # 2. 构造位姿 (保持和 main 一致)
            c2w_src = torch.eye(4, device=self.device).unsqueeze(0)
            # 简单构造一个平移作为目标位姿
            c2w_tgt = torch.eye(4, device=self.device)
            c2w_tgt[0, 3] = 0.2 
            c2w_tgt = c2w_tgt.unsqueeze(0)

            context = {"image": img_tensor, "extrinsics": c2w_src}
            target = {"extrinsics": c2w_tgt}

            # 3. 调用 __call__
            # 这里的 num_inference_steps=1 足够触发编译，节省 warm_up 时间
            try:
                start_warmup = time.time()
                _ = self.__call__(context, target, num_inference_steps=1, debug=False)
                print(f"Warm-up finished in {time.time() - start_warmup:.2f}s.")
            except Exception as e:
                print(f"Warm-up encountered an error, but compilation might have started: {e}")

    @staticmethod
    def split_into_2x2_local_grids(x: torch.Tensor) -> torch.Tensor:
        H, W, C = x.shape
        x = x.view(H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(H // 2, W // 2, 4 * C)
        return x

    def get_closest_resolution(self, H, W):
        aspect_ratio = W / H
        best_res = min(self.preferred_resolutions, key=lambda res: abs(aspect_ratio - res[1] / res[0]))
        return best_res[0], best_res[1]

    def _render_splat(self, src_image, u_tgt, v_tgt, valid_mask, H, W):
        """完全向量化的 Forward Splatting (移除 Python 循环)"""
        B, C, _, _ = src_image.shape
        src_flat = src_image.reshape(B, C, -1)
        
        mask_flat = valid_mask.reshape(-1)
        src_indices = torch.nonzero(mask_flat, as_tuple=True)[0]
        
        u_valid = u_tgt.reshape(-1)[src_indices].long()
        v_valid = v_tgt.reshape(-1)[src_indices].long()
        
        # 边界检查
        in_bounds = (u_valid >= 0) & (u_valid < W) & (v_valid >= 0) & (v_valid < H)
        u_final = u_valid[in_bounds]
        v_final = v_valid[in_bounds]
        src_indices_final = src_indices[in_bounds]
        
        tgt_indices_1d = v_final * W + u_final
        
        # 创建画布并使用批量索引赋值
        tgt_img = torch.zeros((B, C, H * W), device=src_image.device, dtype=src_image.dtype)
        batch_indices = torch.arange(B, device=src_image.device).unsqueeze(1) # [B, 1]
        
        # 向量化赋值：(B, C, N_indices)
        tgt_img[batch_indices, :, tgt_indices_1d] = src_flat[batch_indices, :, src_indices_final]
        
        return tgt_img.view(B, C, H, W)

    def build_pe_field(self, src_image, src_extrinsic, tgt_extrinsic, depth_scale=1.0, debug=False):
        H_orig, W_orig = src_image.shape[1], src_image.shape[2]

        # 1. Moge Inference
        with torch.inference_mode():
            moge_input = src_image.unsqueeze(0).to(self.moge.device)
            moge_output = self.moge.infer(moge_input, resolution_level=9)
            
            depth = moge_output['depth'].squeeze()
            intrinsics_norm = moge_output['intrinsics'] # 保持在 GPU 上

        # [优化] 直接在 Tensor 层面解析内参，避免进入 CPU
        # 假设 Moge 输出为 [1, 3, 3] 或 [3, 3]
        if intrinsics_norm.ndim == 3:
            intrinsics_norm = intrinsics_norm[0]
            
        fx = intrinsics_norm[0, 0] * W_orig
        fy = intrinsics_norm[1, 1] * H_orig
        cx = intrinsics_norm[0, 2] * W_orig
        cy = intrinsics_norm[1, 2] * H_orig

        depth = torch.nan_to_num(depth, posinf=depth[depth.isfinite()].max()) * depth_scale

        # 2. Back-project (使用缓存的 Grid)
        if self._cached_grid[0].shape != (H_orig, W_orig):
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H_orig, device=self.device, dtype=torch.float32),
                torch.arange(W_orig, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            self._cached_grid = (y_grid, x_grid)
        else:
            y_grid, x_grid = self._cached_grid

        # 展平计算以加速
        z_cam = depth.reshape(-1)
        x_cam = (x_grid.reshape(-1) - cx) * z_cam / fx
        y_cam = (y_grid.reshape(-1) - cy) * z_cam / fy
        
        # 3. Transform (Source -> Target)
        # 优化：直接计算 R @ P + T，避免构造庞大的齐次坐标矩阵
        rel_pose = torch.inverse(tgt_extrinsic.float()) @ src_extrinsic.float()
        R = rel_pose[:3, :3]
        T = rel_pose[:3, 3:]

        pts_cam_src = torch.stack([x_cam, y_cam, z_cam], dim=0) # [3, N]
        pts_cam_tgt = R @ pts_cam_src + T # [3, N]
        
        x_tgt, y_tgt, z_tgt = pts_cam_tgt[0], pts_cam_tgt[1], pts_cam_tgt[2]

        # 4. Project
        u_tgt = fx * (x_tgt / (z_tgt + 1e-7)) + cx
        v_tgt = fy * (y_tgt / (z_tgt + 1e-7)) + cy

        # 5. Z-Normalization
        z_min, z_max = z_tgt.min(), z_tgt.max()
        z_norm = (z_tgt - z_min) / (z_max - z_min + 1e-7)

        # 恢复形状
        u_tgt_map = u_tgt.reshape(H_orig, W_orig)
        v_tgt_map = v_tgt.reshape(H_orig, W_orig)
        pix_coords = torch.stack([z_norm.reshape(H_orig, W_orig) + 1.0, u_tgt_map, v_tgt_map], dim=-1)

        # 计算 Warped Image
        valid_mask = z_tgt.reshape(H_orig, W_orig) > 1e-3
        warped_img = self._render_splat(src_image.unsqueeze(0), u_tgt_map, v_tgt_map, valid_mask, H_orig, W_orig)

        # 角度计算 (仅在 B=1 时执行，开销极小)
        phi, theta = 0.0, 0.0
        if debug:
            try:
                forward = R @ torch.tensor([0., 0., -1.], device=self.device)
                forward /= (torch.norm(forward) + 1e-6)
                phi = torch.rad2deg(torch.atan2(forward[0], forward[2])).item()
                phi = phi - 180 if phi > 0 else phi + 180
                theta = torch.rad2deg(torch.asin(torch.clamp(forward[1], -1, 1))).item()
                print(f"Calculate the spherical offset: {phi}, {theta}")
            except:
                pass

        return pix_coords.to(dtype=torch.float32), warped_img.squeeze(0), phi, theta

    def __call__(self, context, target, num_inference_steps=20, depth_scale=1.0, debug=False, **kwargs):
        # 开启推理模式
        with torch.inference_mode():
            src_image = context["image"][0].to(self.device)
            src_extrinsic = context["extrinsics"][0].to(self.device)
            tgt_extrinsic = target["extrinsics"][0].to(self.device)
            
            # 1. Build PE Field
            pix_coords_map, warped_image, phi, theta = self.build_pe_field(
                src_image, src_extrinsic, tgt_extrinsic, depth_scale, debug
            )
            
            H_orig, W_orig = src_image.shape[1], src_image.shape[2]
            H_flux, W_flux = self.get_closest_resolution(H_orig, W_orig)
            
            # 2. Prepare FLUX Input
            src_image_flux = F.interpolate(src_image.unsqueeze(0), size=(H_flux, W_flux), mode='bilinear', align_corners=False)
            src_image_flux = (src_image_flux - 0.5) / 0.5
            
            # 3. Prepare Multi-scale Grid (向量化预处理)
            pix_coords_tensor = pix_coords_map.permute(2, 0, 1).unsqueeze(0)
            H_news_W_news = [(H_flux // 16, W_flux // 16), (H_flux // 8, W_flux // 8), (H_flux // 4, W_flux // 4)]
            
            pix_coords_downs = []
            base_h, base_w = H_news_W_news[0]
            
            for i, (h_new, w_new) in enumerate(H_news_W_news):
                pc_down = F.interpolate(pix_coords_tensor, size=(h_new, w_new), mode='bilinear', align_corners=False)
                pc_down = pc_down.squeeze(0).permute(1, 2, 0) # [H, W, 3]

                pc_down[..., 1] = pc_down[..., 1] / W_orig * base_w
                pc_down[..., 2] = pc_down[..., 2] / H_orig * base_h
                pc_down = pc_down[..., [0, 2, 1]] # z, v, u

                if i > 0:
                    for _ in range(i):
                        pc_down = self.split_into_2x2_local_grids(pc_down)
                
                pix_coords_downs.append(pc_down.reshape(-1, pc_down.shape[-1]).to(dtype=torch.bfloat16))

            # 4. Inference
            output_image = self.pipe(
                image=src_image_flux.to(dtype=torch.bfloat16),
                height=H_flux, width=W_flux,
                prompt="Ensure that the picture does not change in any way",
                input_img_ids=pix_coords_downs,
                use_multi_scale_position=True,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0
            ).images[0]

            # 5. Post-process
            out_tensor = torch.from_numpy(np.array(output_image)).permute(2, 0, 1).float() / 255.0
            if (H_flux != H_orig) or (W_flux != W_orig):
                 out_tensor = F.interpolate(out_tensor.unsqueeze(0), size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze(0)
            else:
                 out_tensor = out_tensor
            
            return out_tensor.unsqueeze(0).to(self.device), phi, theta, warped_image.unsqueeze(0)

# --- 主函数保持不变 ---
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_test")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = PEFieldPipeline(device=device)

    img_pil = Image.open(args.image).convert("RGB")
    W, H = img_pil.size

    new_W, new_H = (W // 32) * 32, (H // 32) * 32 
    img_pil = img_pil.resize((new_W, new_H))
    
    img_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    c2w_src = torch.eye(4)
    theta_rot = np.radians(10)
    c, s = np.cos(theta_rot), np.sin(theta_rot)
    rot_y = torch.tensor([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=torch.float32)
    trans = torch.tensor([[1, 0, 0, 0.2], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]], dtype=torch.float32)
    c2w_tgt = trans @ rot_y @ c2w_src

    context = {"image": img_tensor, "extrinsics": c2w_src.unsqueeze(0)}
    target = {"extrinsics": c2w_tgt.unsqueeze(0)}

    print("Starting optimized inference...")
    start_time = time.time()
    fixed_image, phi, theta, warped_image = pipeline(context, target, num_inference_steps=8)
    print(f"Total Time: {time.time() - start_time:.4f}s")
    
    def save_tensor(t, name):
        img_np = (t.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(args.output, name))

    save_tensor(img_tensor, "input.png")
    save_tensor(warped_image, "warped_debug.png")
    save_tensor(fixed_image, "output_fixed.png")

if __name__ == "__main__":
    main()