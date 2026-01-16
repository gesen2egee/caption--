"""
MainWindow Mixins
"""

from .shortcuts_mixin import ShortcutsMixin
from .theme_mixin import ThemeMixin
from .nl_mixin import NLMixin
from .dialogs_mixin import DialogsMixin
from .progress_mixin import ProgressMixin
from .file_mixin import FileMixin
from .filter_mixin import FilterMixin
from .navigation_mixin import NavigationMixin
from .text_edit_mixin import TextEditMixin

__all__ = ['ShortcutsMixin', 'ThemeMixin', 'NLMixin', 'DialogsMixin', 'ProgressMixin',
           'FileMixin', 'FilterMixin', 'NavigationMixin', 'TextEditMixin']
