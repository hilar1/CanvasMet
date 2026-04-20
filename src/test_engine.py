import numpy as np
import pytest
from engine import calculate_astm_e112, MetallographicEngine

class DummyModelStrategy:
    """为了测试引擎，我们构建一个假的AI策略，它直接返回一张固定的测试图"""
    def predict(self, image_rgb, **kwargs):
        # 创建 100x100 的空白测试掩膜
        mask = np.zeros((100, 100), dtype=np.int32)
        # 绘制 4 个 20x20 的独立晶粒 (注意避开边缘，否则会被 clear_border 清除)
        mask[10:30, 10:30] = 1
        mask[10:30, 70:90] = 2
        mask[70:90, 10:30] = 3
        mask[70:90, 70:90] = 4
        return mask

@pytest.fixture
def test_engine():
    """Pytest 夹具：每次测试提供一个干净的引擎实例"""
    engine = MetallographicEngine(DummyModelStrategy())
    engine.pixel_size_um = 1.0 # 物理标定设为 1 um/px，方便人脑心算验证
    engine.predict(None) # 触发假模型的推断，生成测试掩膜
    return engine

def test_astm_metrics_calculation(test_engine):
    """测试核心业务逻辑：面积与数量计算是否准确"""
    metrics = test_engine.get_all_metrics()
    
    # 我们画了 4 个晶粒，每个都是 20x20 像素
    assert metrics["Valid Grains Count"] == 4
    
    # 物理换算：20px * 1.0um/px = 20um = 0.02mm
    # 面积：0.02 * 0.02 = 0.0004 mm^2
    expected_area_mm2 = 0.0004
    assert np.isclose(metrics["Mean Area (mm^2)"], expected_area_mm2)

def test_delete_roi_interaction(test_engine):
    """测试拓扑交互：点击删除晶粒"""
    # 尝试删除 ID=1 的晶粒 (坐标 20, 20 位于第一个晶粒内部)
    success = test_engine.delete_roi(x=20, y=20)
    assert success is True
    
    # 验证删除后，指标是否实时联动
    metrics = test_engine.get_all_metrics()
    assert metrics["Valid Grains Count"] == 3 # 晶粒数量应该变为 3

def test_add_roi_polygon_interaction(test_engine):
    """测试拓扑交互：多边形绘制生成新晶粒"""
    # 模拟用户在图像中心 (50, 50) 绘制一个 10x10 的正方形
    path = [(45, 45), (55, 45), (55, 55), (45, 55)]
    success = test_engine.add_roi_polygon(path)
    assert success is True
    
    metrics = test_engine.get_all_metrics()
    assert metrics["Valid Grains Count"] == 5 # 晶粒数量应该从 4 增加到 5