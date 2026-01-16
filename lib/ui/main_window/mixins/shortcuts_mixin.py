"""
快捷鍵和事件處理 Mixin

負責處理：
- 快捷鍵設定（圖片導航、刪除等）
- 鍵盤按下事件（N/M 鍵臨時視圖切換）
- 鍵盤釋放事件

依賴的屬性：
- self.temp_view_mode: int | None - 臨時視圖模式
- self.prev_image() - 上一張圖片方法
- self.next_image() - 下一張圖片方法
- self.first_image() - 第一張圖片方法
- self.last_image() - 最後一張圖片方法
- self.delete_current_image() - 刪除當前圖片方法
- self.update_image_display() - 更新圖片顯示方法
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence


class ShortcutsMixin:
    """快捷鍵和事件處理 Mixin"""
    
    def setup_shortcuts(self):
        """
        設定應用程式快捷鍵
        
        快捷鍵列表：
        - Left/Right/PageUp/PageDown: 圖片導航
        - Home/End: 跳到第一張/最後一張
        - Delete: 刪除當前圖片
        - N: 臨時切換到 RGB 視圖（按住時）
        - M: 臨時切換到 Alpha 視圖（按住時）
        """
        # ✅ 圖片左右/翻頁鍵：用 ApplicationShortcut，焦點在 txt 也能翻
        for key, fn in [
            (Qt.Key.Key_Left, self.prev_image),
            (Qt.Key.Key_Right, self.next_image),
            (Qt.Key.Key_PageUp, self.prev_image),
            (Qt.Key.Key_PageDown, self.next_image),
            (Qt.Key.Key_Home, self.first_image),
            (Qt.Key.Key_End, self.last_image),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)

        # Delete 保持原本（避免在 txt 刪字誤觸搬圖）
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.delete_current_image)

    def keyPressEvent(self, event):
        """
        處理鍵盤按下事件
        
        特殊按鍵：
        - N: 臨時切換到 RGB 視圖
        - M: 臨時切換到 Alpha 視圖
        """
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key.Key_N:
            self.temp_view_mode = 1  # RGB
            self.update_image_display()
        elif key == Qt.Key.Key_M:
            self.temp_view_mode = 2  # Alpha
            self.update_image_display()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """
        處理鍵盤釋放事件
        
        當 N 或 M 鍵釋放時，恢復到正常視圖模式
        """
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
            
        key = event.key()
        if key == Qt.Key.Key_N or key == Qt.Key.Key_M:
            # 放開時檢查是否還有其他鍵按著 (簡單起見，直接重置)
            # 如果使用者同時按住 N 和 M，放開一個時會回到 View Mode
            self.temp_view_mode = None
            self.update_image_display()
        else:
            super().keyReleaseEvent(event)
