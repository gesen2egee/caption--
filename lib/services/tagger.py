
import inspect
from PIL import Image
try:
    from imgutils.tagging import get_wd14_tags, remove_underline
except ImportError:
    get_wd14_tags = None
    remove_underline = None

from ..data import AppSettings

def call_wd14(img_pil: Image.Image, settings: AppSettings):
    """
    呼叫 WD14 Tagger (imgutils)。
    自動從 settings 提取參數並過濾不支援的 kwargs。
    """
    if get_wd14_tags is None:
        raise ImportError("imgutils not installed. Please install dghs-imgutils.")

    # 準備參數
    # 優先使用 settings 的屬性，若無則使用 .get 取值
    model_name = settings.get("tagger_model", "EVA02_Large")
    
    # 數值確保為正確型別
    try:
        gen_thresh = float(settings.get("general_threshold", 0.2))
        char_thresh = float(settings.get("character_threshold", 0.85))
    except (ValueError, TypeError):
        gen_thresh = 0.2
        char_thresh = 0.85

    kwargs = {
        "model_name": model_name,
        "general_threshold": gen_thresh,
        "general_mcut_enabled": bool(settings.get("general_mcut_enabled", False)),
        "character_threshold": char_thresh,
        "character_mcut_enabled": bool(settings.get("character_mcut_enabled", True)),
        "drop_overlap": bool(settings.get("drop_overlap", True)),
    }

    # 檢查函數簽章，只傳入支援的參數
    try:
        sig = inspect.signature(get_wd14_tags)
        allowed = set(sig.parameters.keys())
        use = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # Fallback if inspection fails
        use = {
            "model_name": kwargs["model_name"],
            "general_threshold": kwargs["general_threshold"],
            "character_mcut_enabled": kwargs["character_mcut_enabled"],
            "drop_overlap": kwargs["drop_overlap"],
        }

    # 執行模型
    rating, features, chars = get_wd14_tags(img_pil, **use)
    
    # Normalize tags (remove underscores)
    if remove_underline:
        features = {remove_underline(k): v for k, v in features.items()}
        chars = {remove_underline(k): v for k, v in chars.items()}
    else:
        features = {k.replace("_", " "): v for k, v in features.items()}
        chars = {k.replace("_", " "): v for k, v in chars.items()}
    
    return rating, features, chars
