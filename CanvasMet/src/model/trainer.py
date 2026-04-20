import os
import numpy as np
from cellpose import io, models

from src.model.core_train import train_seg

class MetallographicTrainer:
    def __init__(self, gpu=True):
        self.gpu = gpu

    def export_sample(self, image_rgb, mask_array, save_dir, base_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_path = os.path.join(save_dir, f"{base_name}.tif")
        io.imsave(img_path, image_rgb)
        mask_path = os.path.join(save_dir, f"{base_name}_masks.tif")
        io.imsave(mask_path, mask_array.astype(np.uint16))
        return True

    def finetune_model(self, base_model_path, train_dir, epochs=100, learning_rate=0.1, model_name="custom_metal"):
        train_images, train_masks = [], []
        files = os.listdir(train_dir)
        img_files = [f for f in files if f.endswith('.tif') and not f.endswith('_masks.tif')]
        
        for img_name in img_files:
            base_name = img_name.replace('.tif', '')
            mask_name = f"{base_name}_masks.tif"
            if mask_name in files:
                train_images.append(io.imread(os.path.join(train_dir, img_name)))
                train_masks.append(io.imread(os.path.join(train_dir, mask_name)))
                
        if len(train_images) == 0:
            raise ValueError("未在指定目录找到有效的 图像+Mask 训练对！")

        base_model = models.CellposeModel(gpu=self.gpu, pretrained_model=base_model_path)
        
        # 
        if not hasattr(base_model.net, 'device'):
            base_model.net.device = base_model.device

        new_model_path, _, _ = train_seg(
            base_model.net, 
            train_data=train_images, 
            train_labels=train_masks, 
            channels=[0, 0],
            normalize=True,
            save_path=train_dir,   
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=0.0001,
            model_name=model_name
        )
        return new_model_path