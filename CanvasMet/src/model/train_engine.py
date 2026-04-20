import os
import numpy as np
from cellpose import io, models
from core_train import train_seg

class MetallographicTrainer:
    """
    金相模型专用训练引擎。
    负责：1. 训练样本落盘；2. 模型微调 (Fine-tuning)
    """
    def __init__(self, gpu=True):
        self.gpu = gpu

    def export_sample(self, image_rgb, mask_array, save_dir, base_name):
        """
        将当前的原图和修改后的 Mask 导出为 Cellpose 标准训练对
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 保存原图
        img_path = os.path.join(save_dir, f"{base_name}.tif")
        io.imsave(img_path, image_rgb)
        
        # 保存对应的 Mask
        mask_path = os.path.join(save_dir, f"{base_name}_masks.tif")
        io.imsave(mask_path, mask_array.astype(np.uint16))
        
        return True

    def finetune_model(self, base_model_path, train_dir, epochs=100, learning_rate=0.1, model_name="custom_metal"):
        """
        在已有模型基础上，读取训练库数据执行微调
        """
        # 1. 扫描训练文件夹，自动配对原图和 _masks.tif
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

        print(f"开始使用 {len(train_images)} 张标注图像微调模型...")
        
        # 2. 加载基础模型网络
        base_model = models.CellposeModel(gpu=self.gpu, pretrained_model=base_model_path)
        
        # 3. 执行底层的训练循环
        new_model_path, train_losses, test_losses = train_seg(
            base_model.net, 
            train_data=train_images, 
            train_labels=train_masks, 
            channels=[0, 0],       # 保持单通道灰度特征
            normalize=True,
            save_path=train_dir,   
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=0.0001,
            model_name=model_name
        )
        
        return new_model_path