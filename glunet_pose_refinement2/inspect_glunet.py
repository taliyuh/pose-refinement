#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "glunet_src"))

from models.models_compared import GLU_Net

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = GLU_Net(
        path_pre_trained_models=str(ROOT / "glunet_src" / "pre_trained_models"),
        model_type="DPED_CityScape_ADE",
        consensus_network=False,
        cyclic_consistency=True,
        iterative_refinement=True,
        apply_flipping_condition=False,
    )
    model.net = model.net.to(device)
    model.net.eval()
    
    # Load two images
    src_path = ROOT / "images/images_wood/pose_1/image_115723_019413.png"
    tgt_path = ROOT / "images/images_wood/pose_1/image_115750_265186.png"
    
    src = cv2.imread(str(src_path))
    tgt = cv2.imread(str(tgt_path))
    
    # Resize like in run_glunet.py
    max_dim = 1024
    h_src, w_src = src.shape[:2]
    scale_s = max_dim / float(max(h_src, w_src))
    src_resized = cv2.resize(src, (int(w_src * scale_s), int(h_src * scale_s)))
    
    h_tgt, w_tgt = tgt.shape[:2]
    scale_t = max_dim / float(max(h_tgt, w_tgt))
    tgt_resized = cv2.resize(tgt, (int(w_tgt * scale_t), int(h_tgt * scale_t)))
    
    # Check both BGR and RGB
    for mode_name, (img1, img2) in [
        ("BGR (Raw cv2)", (src_resized, tgt_resized)),
        ("RGB (Converted)", (cv2.cvtColor(src_resized, cv2.COLOR_BGR2RGB), cv2.cvtColor(tgt_resized, cv2.COLOR_BGR2RGB)))
    ]:
        print(f"\n--- Running in mode: {mode_name} ---")
        with torch.no_grad():
            source = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
            target = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
            
            flow = model.estimate_flow(source, target, device, mode="channel_first")
            flow_np = flow.squeeze().cpu().numpy()
            
            print(f"Flow shape: {flow_np.shape}")
            print(f"Flow X - Min: {flow_np[0].min():.4f}, Max: {flow_np[0].max():.4f}, Mean: {flow_np[0].mean():.4f}, Std: {flow_np[0].std():.4f}")
            print(f"Flow Y - Min: {flow_np[1].min():.4f}, Max: {flow_np[1].max():.4f}, Mean: {flow_np[1].mean():.4f}, Std: {flow_np[1].std():.4f}")
            
            # Let's count how many pixels have non-zero displacement (e.g. > 1 px)
            magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            active_px = np.sum(magnitude > 1.0)
            total_px = magnitude.size
            print(f"Pixels with displacement > 1.0 px: {active_px} / {total_px} ({active_px/total_px*100:.2f}%)")

if __name__ == "__main__":
    main()
