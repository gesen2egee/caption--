"""
MainWindow Mixins

這個模組包含 MainWindow 的所有 Mixin 類別，每個 Mixin 負責特定的功能領域。

Mixin 設計原則：
1. 每個 Mixin 只負責一個功能領域
2. Mixin 之間通過共享屬性通信
3. 所有共享屬性在 MainWindow.__init__() 中初始化
"""

from .shortcuts_mixin import ShortcutsMixin
from .theme_mixin import ThemeMixin
from .nl_mixin import NLMixin

__all__ = ['ShortcutsMixin', 'ThemeMixin', 'NLMixin']
