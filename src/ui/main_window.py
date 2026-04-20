import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QFrame, QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QGroupBox, QLineEdit, QCheckBox, QSpinBox, 
                               QFormLayout, QGridLayout)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtCore import Qt

# ===== 从内部引擎层导入 =====
from src.engine.core_engine import MetallographicEngine
from src.model.option import CellposeStrategy
from src.model.trainer import MetallographicTrainer
from src.ui.viewport import ImageViewport

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("金相智能分析系统 v1.0 - Enterprise Architecture")
        self.resize(1300, 950)
        # self.base_path = "models"
        # self.custom_name = "custom_metal"
        
        default_model = 'models/CP_20260324_4'
        if os.path.exists(default_model):
            self.engine = MetallographicEngine(CellposeStrategy(default_model))
            # 1. 隐藏保存完整路径
            self.current_model_full_path = default_model 
            # 2. 提取文件名用于 UI 显示
            self.display_model_name = os.path.basename(default_model) 
        else:
            self.engine = MetallographicEngine(None)
            self.current_model_full_path = ""
            self.display_model_name = "未检测到模型..."
        
        self.trainer = MetallographicTrainer(gpu=True)
        self.setup_ui()
        self.connect_signals()
        self.sync_ui_state()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        control_panel = QVBoxLayout()
        
        # [0. 模型区]
        model_group = QGroupBox("模型配置")
        model_layout = QHBoxLayout(model_group)
        self.input_model_path = QLineEdit(self.display_model_name)
        self.input_model_path.setReadOnly(True) 
        self.btn_choose_model = QPushButton("浏览...")
        model_layout.addWidget(self.input_model_path)
        model_layout.addWidget(self.btn_choose_model)
        control_panel.addWidget(model_group)

        # [1. 操作区]
        op_group = QGroupBox("基础操作")
        op_layout = QGridLayout(op_group)
        self.btn_load = QPushButton("导入图像")
        self.btn_open_proj = QPushButton("打开工程")
        self.btn_save_proj = QPushButton("保存工程")
        self.btn_predict = QPushButton("智能识别")
        op_layout.addWidget(self.btn_load, 0, 0)
        op_layout.addWidget(self.btn_open_proj, 0, 1)
        op_layout.addWidget(self.btn_save_proj, 1, 0)
        op_layout.addWidget(self.btn_predict, 1, 1)
        control_panel.addWidget(op_group)

        # [2. 训练区]
        training_group = QGroupBox("模型迭代")
        training_layout = QVBoxLayout(training_group)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("新模型命名:"))
        self.input_new_model_name = QLineEdit("custom_metal")
        self.btn_export_sample = QPushButton("导出为训练样本")
        self.btn_finetune = QPushButton("选择样本库并微调")
        name_layout.addWidget(self.input_new_model_name)
        training_layout.addLayout(name_layout)
        training_layout.addWidget(self.btn_export_sample)
        training_layout.addWidget(self.btn_finetune)
        control_panel.addWidget(training_group)

        # [3. 渲染区]
        render_group = QGroupBox("图层控制")
        render_layout = QFormLayout(render_group)
        self.chk_show_mask = QCheckBox("显示晶粒 Mask"); self.chk_show_mask.setChecked(True)
        self.chk_show_outline = QCheckBox("显示晶粒边界线"); self.chk_show_outline.setChecked(True)
        self.spin_outline_width = QSpinBox()
        self.spin_outline_width.setRange(2, 10); self.spin_outline_width.setValue(2)
        render_layout.addRow(self.chk_show_mask); render_layout.addRow(self.chk_show_outline)
        render_layout.addRow("线宽:", self.spin_outline_width)
        control_panel.addWidget(render_group)

        # [4. 报告区]
        data_group = QGroupBox("实时报告")
        data_layout = QGridLayout(data_group)
        self.input_pixel = QLineEdit("0.438")
        self.lbl_count = QLabel("有效晶粒: 0")
        self.lbl_astm = QLabel("晶粒度(G): 0.0")
        self.lbl_intercept = QLabel("平均截距: 0.0 μm")
        data_layout.addWidget(QLabel("换算率(μm/px):"), 0, 0)
        data_layout.addWidget(self.input_pixel, 0, 1)
        data_layout.addWidget(self.lbl_count, 1, 0)
        data_layout.addWidget(self.lbl_astm, 1, 1)
        data_layout.addWidget(self.lbl_intercept, 2, 0, 1, 2)
        control_panel.addWidget(data_group)
        
        self.lbl_tips = QLabel("- 左键单击: 删除 | Ctrl+拖拽: 绘制")
        self.lbl_tips.setStyleSheet("color: gray;")
        control_panel.addWidget(self.lbl_tips)
        control_panel.addStretch()
        
        # [加载视口]
        self.viewport = ImageViewport()
        main_layout.addLayout(control_panel, 1)
        main_layout.addWidget(self.viewport, 4)

        # [悬浮按钮]
        # 1. 创建胶囊底座容器，父类设为 viewport 即可悬浮
        self.floating_bar = QFrame(self.viewport)
        self.floating_bar.setObjectName("FloatingBar")
        
        # 2. 为底座设置水平布局
        floating_layout = QHBoxLayout(self.floating_bar)
        floating_layout.setContentsMargins(6, 6, 6, 6)  # 容器内边距
        floating_layout.setSpacing(10)                  # 按钮之间的硬性安全间距，绝对防止重叠
        
        # 3. 创建按钮
        self.btn_undo = QPushButton("↶")
        self.btn_redo = QPushButton("↷")
        self.btn_undo.setObjectName("FloatingBtn")
        self.btn_redo.setObjectName("FloatingBtn")
        
        self.btn_undo.setToolTip("撤销 (Ctrl+Z)")
        self.btn_redo.setToolTip("重做 (Ctrl+Y)")
        
        # 4. 将按钮加入布局
        floating_layout.addWidget(self.btn_undo)
        floating_layout.addWidget(self.btn_redo)
        
        # 5. 读取QSS样式文件
        try:
            with open(os.path.join(os.path.dirname(__file__), 'style.qss'), 'r') as f:
                self.setStyleSheet(f.read()) # 将样式表应用到全局，QFrame和QPushButton会自动拾取
        except Exception as e:
            print(f"样式加载警告: {e}")

        # 6. 仅需移动整体容器即可，内部按钮位置由 layout 自动计算
        self.floating_bar.move(20, 20)
        
        # 读取QSS样式文件
        try:
            with open(os.path.join(os.path.dirname(__file__), 'style.qss'), 'r') as f:
                style = f.read()
                self.btn_undo.setStyleSheet(style)
                self.btn_redo.setStyleSheet(style)
        except:
            pass # 样式加载失败不影响运行

        self.btn_undo.move(15, 15)
        self.btn_redo.move(55, 15)

    def connect_signals(self):
        self.btn_choose_model.clicked.connect(self.choose_model)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_open_proj.clicked.connect(self.load_project_file)
        self.btn_save_proj.clicked.connect(self.save_project_file)
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_export_sample.clicked.connect(self.export_sample)
        self.btn_finetune.clicked.connect(self.start_finetuning)
        self.btn_undo.clicked.connect(self.exec_undo)
        self.btn_redo.clicked.connect(self.exec_redo)
        
        shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut_undo.setContext(Qt.ApplicationShortcut)
        shortcut_undo.activated.connect(self.exec_undo)

        shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        shortcut_redo.setContext(Qt.ApplicationShortcut)
        shortcut_redo.activated.connect(self.exec_redo)

        self.input_pixel.textChanged.connect(self.sync_ui_state)
        self.viewport.sig_roi_clicked.connect(self.handle_roi_click)
        self.viewport.sig_draw_finished.connect(self.handle_draw_finished)
        self.chk_show_mask.stateChanged.connect(self.sync_ui_state)
        self.chk_show_outline.stateChanged.connect(self.sync_ui_state)
        self.spin_outline_width.valueChanged.connect(self.sync_ui_state)

    def exec_undo(self):
        if self.engine.undo(): self.sync_ui_state()

    def exec_redo(self):
        if self.engine.redo(): self.sync_ui_state()

    def sync_ui_state(self):
        self.viewport.render_overlays(
            mask_array=self.engine.current_mask,
            outline_width=self.spin_outline_width.value(),
            show_mask=self.chk_show_mask.isChecked(),
            show_outline=self.chk_show_outline.isChecked()
        )
        try: self.engine.pixel_size_um = float(self.input_pixel.text())
        except ValueError: pass
            
        metrics = self.engine.get_all_metrics()
        if not metrics:
            self.lbl_count.setText("有效: 0"); self.lbl_astm.setText("G: 0.0"); self.lbl_intercept.setText("截距: 0.0 μm")
        else:
            self.lbl_count.setText(f"有效: {metrics['Valid Grains Count']}")
            self.lbl_astm.setText(f"G: {metrics['ASTM Grain Size (G)']:.2f}")
            self.lbl_intercept.setText(f"截距: {metrics['Mean Intercept (um)']:.2f} μm")

        history_mgr = self.engine.history_mgr
        self.btn_undo.setEnabled(history_mgr.current_step >= 0)
        self.btn_redo.setEnabled(history_mgr.current_step < len(history_mgr.history) - 1)

        has_img = self.engine.current_image is not None
        self.btn_predict.setEnabled(has_img and self.engine.ai_strategy is not None)
        self.btn_save_proj.setEnabled(has_img)

    def handle_roi_click(self, x, y):
        if self.engine.delete_roi(x, y): self.sync_ui_state()

    def handle_draw_finished(self, path_points):
        if self.engine.add_roi_polygon(path_points): self.sync_ui_state()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.tif)")
        if path:
            img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            self.engine.current_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.engine.current_mask = None
            self.engine.history_mgr.history.clear() 
            self.engine.history_mgr.current_step = -1
            self.viewport.set_image(self.engine.current_image)
            self.sync_ui_state()

    def run_prediction(self):
        self.btn_predict.setText("识别中..."); self.btn_predict.setEnabled(False)
        QApplication.processEvents() 
        self.engine.predict(self.engine.current_image)
        self.btn_predict.setText("智能识别")
        self.sync_ui_state()

    def save_project_file(self):
        if self.engine.current_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "保存工程", "", "Metaproj (*.metaproj)")
        if path:
            if not path.endswith('.metaproj'): path += '.metaproj'
            if self.engine.save_project(path): self.lbl_tips.setText(f"已保存: {os.path.basename(path)}")

    def load_project_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Metaproj (*.metaproj)")
        if path:
            try:
                self.engine.load_project(path)
                self.viewport.set_image(self.engine.current_image)
                self.input_pixel.setText(str(self.engine.pixel_size_um))
                self.engine.history_mgr.history.clear() 
                self.engine.history_mgr.current_step = -1
                if self.engine.current_mask is not None:
                    self.engine.history_mgr.init_base(self.engine.current_mask)
                self.sync_ui_state()
            except Exception as e:
                self.lbl_tips.setText(f"加载失败: {str(e)}")

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "All (*)")
        if path:
            # 1. 保存完整路径
            self.current_model_full_path = path
            # 2. 提取并显示文件名
            model_name = os.path.basename(path)
            
            self.input_model_path.setText(model_name)
            self.input_model_path.setStyleSheet("color: #00FF00;")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try: 
                # 后台依然使用完整路径实例化引擎
                self.engine.ai_strategy = CellposeStrategy(self.current_model_full_path)
            except: 
                self.input_model_path.setText("加载失败！")
                self.input_model_path.setStyleSheet("color: red;")
                self.engine.ai_strategy = None
            finally:
                QApplication.restoreOverrideCursor()
                self.sync_ui_state()

    def export_sample(self):
        if self.engine.current_image is None or self.engine.current_mask is None: return
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if dir_path:
            import time
            base_name = f"sample_{int(time.time())}"
            if self.trainer.export_sample(self.engine.current_image, self.engine.current_mask, dir_path, base_name):
                self.lbl_tips.setText(f"已导出: {base_name}")

    def start_finetuning(self):
        if not self.engine.ai_strategy: return
        dir_path = QFileDialog.getExistingDirectory(self, "选择样本库")
        if dir_path:
            self.btn_finetune.setText("训练中..."); self.btn_finetune.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            # try:
                # new_path = self.trainer.finetune_model(self.base_path, dir_path, epochs=100, model_name=self.custom_name)
                # if new_path:
                #     new_path_str = str(new_path)
                    
                #     # 1. 记录后台完整路径
                #     self.current_model_full_path = new_path_str
                #     # 2. 提取仅供显示的名称
                #     display_name = os.path.basename(new_path_str)
                    
                #     self.input_model_path.setText(display_name)
                #     self.engine.ai_strategy = CellposeStrategy(new_path_str)
                #     self.lbl_tips.setText(f"已加载: {self.custom_name}")
            try:
                raw_path = self.engine.ai_strategy.model.pretrained_model
                base_path = str(raw_path[0] if isinstance(raw_path, list) else raw_path)
                
                custom_name = self.input_new_model_name.text().strip() or "custom_metal"

                new_path = self.trainer.finetune_model(
                    base_path, 
                    dir_path, 
                    epochs=100, 
                    model_name=custom_name
                )
                
                if new_path:
                    new_path_str = str(new_path)
                    
                    # 1. 记录后台完整路径
                    self.current_model_full_path = new_path_str
                    # 2. 提取仅供显示的名称
                    display_name = os.path.basename(new_path_str)
                    
                    # 3. 驱动 UI 视图层与算法模型层同步更新
                    self.input_model_path.setText(display_name)
                    self.engine.ai_strategy = CellposeStrategy(new_path_str)
                    self.lbl_tips.setText(f"已加载: {custom_name}")
                    
            except Exception as e:
                self.lbl_tips.setText(f"异常: {str(e)}")
            finally:
                self.btn_finetune.setText("选择样本库并微调"); self.btn_finetune.setEnabled(True)
                QApplication.restoreOverrideCursor()
                self.sync_ui_state()