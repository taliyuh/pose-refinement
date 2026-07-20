import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add workspace and glunet paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "glunet_pose_refinement2"))
sys.path.insert(0, str(ROOT / "glunet_src"))

from estimate_glunet import (
    load_glunet,
    parse_gt,
    evaluate_pair
)

def run_refinement_on_all_depth_photos():
    images_dir = ROOT / "images"
    depth_dir = images_dir / "images_depth"
    gt_path = images_dir / "ground truth.txt"
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    gt = parse_gt(gt_path)
    
    # Correct the ground-truth typo if present
    if "image_115740_590015.png" in gt:
        gt["image_115740_590015.png"][3, 0] = 2685.0
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading GLU-Net model...")
    model = load_glunet(device)
    print("Model loaded successfully.")
    
    # Gather pose folders: pose_1, pose_2, pose_3
    pose_folders = sorted([d for d in depth_dir.iterdir() if d.is_dir() and d.name.startswith("pose_")])
    
    for p_folder in pose_folders:
        print(f"\n==================== PROCESSING {p_folder.name} ====================")
        # Gather all timestamped subdirectories
        subdirs = sorted([d for d in p_folder.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"No subdirectories found in {p_folder.name}. Skipping.")
            continue
            
        # Treat the first photo folder as the source
        src_dir = subdirs[0]
        src_path = src_dir / "color.png"
        print(f"Source image: {src_dir.name}/color.png")
        
        # All other photo folders are targets
        targets = subdirs[1:]
        print(f"Found {len(targets)} targets to evaluate.")
        
        for tgt_dir in tqdm(targets, desc=f"Evaluating {p_folder.name} targets"):
            tgt_path = tgt_dir / "color.png"
            evaluate_pair(model, src_path, tgt_path, gt, device, out_dir)

if __name__ == "__main__":
    run_refinement_on_all_depth_photos()
