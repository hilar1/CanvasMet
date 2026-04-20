import cv2
import numpy as np
import zipfile
import json
import io

from src.model.option import BaseAIModel
from src.engine.op_log import MaskHistoryManager
from src.engine.metrics import METRICS_REGISTRY

class MetallographicEngine:
    def __init__(self, ai_strategy: BaseAIModel):
        self.ai_strategy = ai_strategy
        self.current_image = None
        self.current_mask = None
        self.pixel_size_um = 0.438
        self.history_mgr = MaskHistoryManager(max_steps=50) 

    def predict(self, image_rgb, **kwargs):
        self.current_image = image_rgb
        self.current_mask = self.ai_strategy.predict(image_rgb, **kwargs)
        self.history_mgr.init_base(self.current_mask)
        return self.current_mask

    def get_all_metrics(self):
        if self.current_mask is None: return None
        aggregated_metrics = {}
        for metric_func in METRICS_REGISTRY:
            aggregated_metrics.update(metric_func(self.current_mask, self.pixel_size_um))
        return aggregated_metrics

    def delete_roi(self, x, y):
        if self.current_mask is None: return False
        height, width = self.current_mask.shape
        if y < 0 or x < 0 or y >= height or x >= width: return False
        
        target_id = self.current_mask[y, x]
        if target_id > 0:
            old_mask = self.current_mask.copy() 
            self.current_mask[self.current_mask == target_id] = 0
            self.history_mgr.push(old_mask, self.current_mask, f"删除晶粒 (ID: {target_id})")
            return True
        return False

    def add_roi_polygon(self, path_points):
        if self.current_mask is None or len(path_points) < 3: return False
        old_mask = self.current_mask.copy()
        pts = np.array(path_points, np.int32).reshape((-1, 1, 2))
        new_id = np.max(self.current_mask) + 1
        cv2.fillPoly(self.current_mask, [pts], color=int(new_id))
        self.history_mgr.push(old_mask, self.current_mask, "手动绘制新晶粒")
        return True

    def undo(self):
        return self.history_mgr.undo(self.current_mask)

    def redo(self):
        return self.history_mgr.redo(self.current_mask)

    def save_project(self, file_path):
        if self.current_image is None: return False
        metadata = {
            "pixel_size_um": self.pixel_size_um,
            "software_version": "7.0",
            "model_type": self.ai_strategy.__class__.__name__ if self.ai_strategy else "None"
        }
        with zipfile.ZipFile(file_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('meta.json', json.dumps(metadata, indent=4))
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
            zf.writestr('image.png', img_encoded.tobytes())
            if self.current_mask is not None:
                mask_buffer = io.BytesIO()
                np.save(mask_buffer, self.current_mask)
                zf.writestr('mask.npy', mask_buffer.getvalue())
        return True

    def load_project(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zf:
            with zf.open('meta.json') as f:
                metadata = json.loads(f.read())
                self.pixel_size_um = metadata.get("pixel_size_um", 0.438)

            with zf.open('image.png') as f:
                img_array = np.frombuffer(f.read(), np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.current_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if 'mask.npy' in zf.namelist():
                with zf.open('mask.npy') as f:
                    self.current_mask = np.load(io.BytesIO(f.read()))
            else:
                self.current_mask = None
        return self.current_image, self.current_mask