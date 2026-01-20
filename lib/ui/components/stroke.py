# -*- coding: utf-8 -*-
"""
手繪橡皮擦工具組件
"""
from PyQt6.QtWidgets import (
    QLabel, QDialog, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QWidget
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor
)
from PyQt6.QtCore import Qt, QPoint
from lib.locales import tr as _tr


def create_checkerboard_png_bytes(size=16, color1=(200, 200, 200), color2=(150, 150, 150)):
    """Create a checkerboard pattern as PNG bytes for use as background."""
    from PyQt6.QtGui import QPainter, QColor
    from PyQt6.QtCore import QBuffer, QIODevice, QByteArray
    
    img = QImage(size * 2, size * 2, QImage.Format.Format_RGB888)
    painter = QPainter(img)
    
    # Draw checkerboard
    painter.fillRect(0, 0, size, size, QColor(*color1))
    painter.fillRect(size, 0, size, size, QColor(*color2))
    painter.fillRect(0, size, size, size, QColor(*color2))
    painter.fillRect(size, size, size, size, QColor(*color1))
    painter.end()
    
    # Convert to PNG bytes
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    
    return ba.data()


class StrokeCanvas(QLabel):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.base_pixmap = pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mask = QImage(self.base_pixmap.size(), QImage.Format.Format_Grayscale8)
        self.mask.fill(0)

        self.preview = QPixmap(self.base_pixmap.size())
        self.preview.fill(Qt.GlobalColor.transparent)

        self.pen_width = 30
        self.drawing = False
        self.last_pos = None

        self._update_display()

    def set_pen_width(self, w: int):
        self.pen_width = max(1, int(w))

    def clear_mask(self):
        self.mask.fill(0)
        self.preview.fill(Qt.GlobalColor.transparent)
        self._update_display()

    def _draw_line(self, p1, p2):
        # draw to mask
        painter = QPainter(self.mask)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(self.pen_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()

        # draw preview overlay
        painter2 = QPainter(self.preview)
        pen2 = QPen(QColor(255, 0, 0, 160))
        pen2.setWidth(self.pen_width)
        pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen2.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter2.setPen(pen2)
        painter2.drawLine(p1, p2)
        painter2.end()

    def _update_display(self):
        pm = QPixmap(self.base_pixmap)
        painter = QPainter(pm)
        painter.drawPixmap(0, 0, self.preview)
        painter.end()
        self.setPixmap(pm)

    def _to_image_pos(self, widget_pos: QPoint):
        """Map a widget (label) position to image pixel coords, respecting centered pixmap."""
        pm_w = self.base_pixmap.width()
        pm_h = self.base_pixmap.height()
        off_x = int((self.width() - pm_w) / 2)
        off_y = int((self.height() - pm_h) / 2)
        x = int(widget_pos.x() - off_x)
        y = int(widget_pos.y() - off_y)
        if x < 0 or y < 0 or x >= pm_w or y >= pm_h:
            return None
        return QPoint(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            p = self._to_image_pos(event.position().toPoint())
            if p is None:
                event.ignore()
                return
            self.drawing = True
            self.last_pos = p
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            p = self._to_image_pos(event.position().toPoint())
            if p is None:
                event.ignore()
                return
            if self.last_pos is not None:
                self._draw_line(self.last_pos, p)
                self.last_pos = p
                self._update_display()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def get_mask(self) -> QImage:
        return QImage(self.mask)


class StrokeEraseDialog(QDialog):
    def tr(self, key: str) -> str:
        lang = "zh_tw"
        if self.parent() and hasattr(self.parent(), "settings"):
            lang = self.parent().settings.get("ui_language", "zh_tw")
        load_locale(lang)
        return _tr(key)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("title_stroke_eraser"))
        self.image_path = image_path
        self._mask = None

        layout = QVBoxLayout(self)

        # load image (fit to a reasonable size)
        pm = QPixmap(image_path)
        if pm.isNull():
            raise RuntimeError("Cannot load image")

        max_w, max_h = 1200, 800
        if pm.width() > max_w or pm.height() > max_h:
            pm = pm.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.canvas = StrokeCanvas(pm)
        layout.addWidget(self.canvas, 1)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel(self.tr("label_pen_width")))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(120)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(lambda v: self.canvas.set_pen_width(v))
        ctrl.addWidget(self.slider, 1)

        self.btn_clear = QPushButton(self.tr("btn_clear_action"))
        self.btn_clear.clicked.connect(self.canvas.clear_mask)
        ctrl.addWidget(self.btn_clear)

        layout.addLayout(ctrl)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_apply = QPushButton(self.tr("btn_apply_action"))
        self.btn_cancel = QPushButton(self.tr("setting_cancel"))
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

    def get_result(self):
        return self.canvas.get_mask(), int(self.slider.value())
