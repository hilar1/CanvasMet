import numpy as np
from abc import ABC, abstractmethod
from cellpose import models

class BaseAIModel(ABC):
    @abstractmethod
    def predict(self, image_rgb, **kwargs) -> np.ndarray:
        pass

class CellposeStrategy(BaseAIModel):
    def __init__(self, model_path, gpu=True):
        self.model_path = model_path
        self.model = models.CellposeModel(gpu=gpu, pretrained_model=model_path)

    def predict(self, image_rgb, diameter=None, **kwargs):
        masks, _, _ = self.model.eval(image_rgb, diameter=diameter, channels=[0, 0])
        return masks