"""
MainWindow Mixins

這個模組包含 MainWindow 的所有 Mixin 類別，每個 Mixin 負責特定的功能領域。

Mixin 設計原則：
1. 每個 Mixin 只負責一個功能領域
2. Mixin 之間通過共享屬性通信
3. 所有共享屬性在 MainWindow.__init__() 中初始化
"""

# 將在後續 Phase 中添加各個 Mixin 的導入
__all__ = []
