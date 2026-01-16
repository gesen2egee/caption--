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
from .tags_mixin import TagsMixin
from .image_mixin import ImageMixin
from .batch_base_mixin import BatchBaseMixin
from .vision_mixin import VisionMixin
from .tagger_mixin import TaggerMixin
from .llm_mixin import LLMMixin
from .app_core_mixin import AppCoreMixin

__all__ = ['ShortcutsMixin', 'ThemeMixin', 'NLMixin', 'DialogsMixin', 'ProgressMixin',
           'FileMixin', 'FilterMixin', 'NavigationMixin', 'TextEditMixin', 'TagsMixin',
           'ImageMixin', 'BatchBaseMixin', 'VisionMixin', 'TaggerMixin', 'LLMMixin',
           'AppCoreMixin']
