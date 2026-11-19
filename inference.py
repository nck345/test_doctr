from GeoTr import GeoTr

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
from PIL import Image
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GeoTrP(nn.Module):
    def __init__(self):
        super(GeoTrP, self).__init__()
        try:
            from GeoTr_orig import GeoTr as GeoTrModel
            self.GeoTr = GeoTrModel(num_attn_layers=6)
        except Exception:
            self.GeoTr = GeoTr()
        
    def forward(self, x):
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)
        print(f"Loading weights from {path}: total keys = {len(pretrained_dict.keys())}")
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(f"Matched keys in model = {len(pretrained_dict.keys())}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

def process_folder(GeoTrP_model, input_dir, output_dir, benchmark_name=''):
    os.makedirs(output_dir, exist_ok=True)
    img_list = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if not img_list:
        print(f"No images found in {input_dir}")
        return

    tag = benchmark_name if benchmark_name else os.path.basename(output_dir)
    t0 = time.time()
    
    pbar = tqdm(img_list, desc=f"{tag:<20}", unit="img", ncols=95)
    for img_path in pbar:
        name = os.path.splitext(img_path)[0]
        full_img_path = os.path.join(input_dir, img_path)
        im_ori = np.array(Image.open(full_img_path))[:, :, :3] / 255. 
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            bm = GeoTrP_model(im)
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).to(device)
            
            im_tensor = torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float().to(device)
            out = F.grid_sample(im_tensor, lbl, align_corners=True)
            img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy())[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, name + '_geo.png'), img_geo)

        if device.type == 'cuda':
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            pbar.set_postfix({'VRAM': f"{vram_mb:.0f}MB"})

    elapsed = time.time() - t0
    fps = len(img_list) / max(elapsed, 1e-5)
    print(f"✅ Hoàn tất [{tag}]: {len(img_list)} ảnh trong {elapsed:.1f}s ({fps:.1f} img/s)\n")

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--distorrted_path', default='./distorted/')
    parser.add_argument('--gsave_path', default='./rectified/')
    parser.add_argument('--GeoTr_path', default='./model_pretrained/DocTrP.pth')
    parser.add_argument('--base_dir', default=None, help='Base directory of Document_Dewarp_Benchmark')
    parser.add_argument('--benchmarks', nargs='+', default=None, help='List of benchmarks or all')
    
    opt = parser.parse_args()

    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    GeoTrP_model = GeoTrP().to(device)
    reload_model(GeoTrP_model.GeoTr, opt.GeoTr_path)
    GeoTrP_model.eval()

    if opt.base_dir:
        all_bm = ['docunet_paper_subset', 'docunet_130', 'dir300', 'docreal']
        if not opt.benchmarks or 'all' in opt.benchmarks:
            target_bms = all_bm
        else:
            target_bms = opt.benchmarks

        print(f"\n🚀 Bắt đầu nắn phẳng {len(target_bms)} tập benchmark: {target_bms}")
        for bm in target_bms:
            in_dir = os.path.join(opt.base_dir, 'benchmarks', bm, 'crop')
            if not os.path.exists(in_dir):
                in_dir = os.path.join(opt.base_dir, 'data_benchmark', 'crop', 'crop')
            out_dir = os.path.join(opt.base_dir, 'predictions', 'DocTr', bm)
            process_folder(GeoTrP_model, in_dir, out_dir, benchmark_name=bm)
    else:
        process_folder(GeoTrP_model, opt.distorrted_path, opt.gsave_path)

if __name__ == '__main__':
    main()
