import os
import torch
import numpy as np
from PIL import Image
from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
model_path = "/mnt/petrelfs/zhouyujie/hwfile/ckpt/sam2/sam2_b.pt"
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=model_path)
predictor = SAM2VideoPredictor(overrides=overrides)

video_name = "tabby_cat"
results = predictor(source=f"input_animatediff/{video_name}.mp4",points=[255, 255],labels=[1])

for i in range(len(results)):
    mask = (results[i].masks.data).squeeze().to(torch.float16)
    mask = (mask * 255).cpu().numpy().astype(np.uint8)
    mask_image = Image.fromarray(mask)
    mask_dir = f'masks_animatediff/{video_name}'
    if not os.path.exists(mask_dir):  
        os.makedirs(mask_dir)        
    mask_image.save(mask_dir + f'/{str(i).zfill(3)}.png')
    