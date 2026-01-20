# -*- coding: utf-8 -*-
"""
Caption 神器 - 核心模組
"""
from lib.core.settings import (
    DEFAULT_APP_SETTINGS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_CUSTOM_TAGS,
    load_app_settings,
    save_app_settings,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
)

from lib.core.dataclasses import (
    ImageData,
    Settings,
    Prompt,
    BatchInstruction,
    FolderMeta,
)
