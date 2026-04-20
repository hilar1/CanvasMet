import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
import cv2
from skimage.segmentation import find_boundaries

class InteractiveMaskItem(pg.ImageItem):
    sig_draw_finished = QtCore.Signal(list)
    sig_roi_clicked = QtCore.Signal(int, int)

    def __init__(self, viewbox):
        super().__init__()
        self.viewbox = viewbox
        self.drawing = False
        self.current_path = []
        
        # 红色轨迹线图层 (Z值设为极高)
        self.draw_curve = pg.PlotCurveItem(pen=pg.mkPen('r', width=2))
        self.viewbox.addItem(self.draw_curve)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and ev.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier:
            pos = ev.pos()
            self.sig_roi_clicked.emit(int(pos.x()), int(pos.y()))
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev):
        if ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            if ev.isStart():
                self.drawing = True
                self.current_path = []
            
            if self.drawing:
                pos = ev.pos()
                px, py = int(pos.x()), int(pos.y())
                
                # 防抖：避免鼠标未移动时加入重复点
                if len(self.current_path) == 0 or self.current_path[-1] != (px, py):
                    self.current_path.append((px, py))
                    self.draw_curve.setData([p[0] for p in self.current_path], 
                                            [p[1] for p in self.current_path])
                    
                    # 停止逻辑一：检测轨迹自交 (首尾相连或画圈交叉)
                    # 保留 15 个点的缓冲带，防止刚画出就被判定为与起点相交
                    if len(self.current_path) > 15:
                        prev_pts = np.array(self.current_path[:-15])
                        curr_pt = np.array([px, py])
                        # 计算平方距离以提升执行速度
                        sq_dists = np.sum((prev_pts - curr_pt)**2, axis=1)
                        if np.any(sq_dists <= 4):  # 阈值：距离在2个像素以内判定为相交
                            self.drawing = False
                            self.sig_draw_finished.emit(self.current_path)
                            self.draw_curve.clear()
                            self.current_path = []
                            ev.accept()
                            return # 触发闭合，提前退出
            
            # 停止逻辑二：用户松开鼠标左键
            if ev.isFinish() and self.drawing:
                self.drawing = False
                self.sig_draw_finished.emit(self.current_path)
                self.draw_curve.clear()
                self.current_path = []
            
            ev.accept()
        else:
            ev.ignore()


class ImageViewport(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.view = self.addViewBox(lockAspect=True, enableMenu=False, invertY=True)
        
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)
        
        self.mask_item = InteractiveMaskItem(self.view)
        self.mask_item.setZValue(10) 
        self.view.addItem(self.mask_item)

        self.outline_item = pg.ImageItem()
        self.outline_item.setZValue(20)
        self.view.addItem(self.outline_item)

        self.color_map = None

    @property
    def sig_draw_finished(self): return self.mask_item.sig_draw_finished
    @property
    def sig_roi_clicked(self): return self.mask_item.sig_roi_clicked

    def set_image(self, img_rgb):
        self.img_item.setImage(np.swapaxes(img_rgb, 0, 1), autoLevels=True)
        self.view.autoRange()

    def set_layers_visibility(self, show_mask, show_outline):
        self.mask_item.setVisible(show_mask)
        self.outline_item.setVisible(show_outline)

    def render_overlays(self, mask_array, outline_width=2, show_mask=True, show_outline=True):
        if mask_array is None: return

        # Mask 层渲染
        max_id = np.max(mask_array)
        if self.color_map is None or len(self.color_map) <= max_id:
            self.color_map = np.random.randint(0, 255, size=(max_id + 1000, 4), dtype=np.uint8)
            self.color_map[:, 3] = 100 
            self.color_map[0] = [0, 0, 0, 0] 

        rgba_mask = self.color_map[mask_array]
        self.mask_item.setImage(np.swapaxes(rgba_mask, 0, 1), autoLevels=False)

        # 边界层渲染
        bound = find_boundaries(mask_array, mode='thick').astype(np.uint8)
        if outline_width > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_width, outline_width))
            bound = cv2.dilate(bound, kernel)
        
        rgba_outline = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        rgba_outline[bound > 0] = [255, 255, 0, 255] 
        self.outline_item.setImage(np.swapaxes(rgba_outline, 0, 1), autoLevels=False)

        self.set_layers_visibility(show_mask, show_outline)