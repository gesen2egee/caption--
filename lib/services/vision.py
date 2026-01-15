
import numpy as np
from PIL import Image, ImageFilter
from ..data import AppSettings

# Optional imports
try:
    from transparent_background import Remover
except ImportError:
    Remover = None

try:
    from imgutils.ocr import detect_text_with_ocr
except ImportError:
    detect_text_with_ocr = None

# ==========================================
#  Background Removal
# ==========================================

def get_remover_model(settings: AppSettings):
    """
    Factory to create or retrieve the Remover instance.
    Caller is responsible for caching this if needed.
    """
    if Remover is None:
        raise ImportError("transparent_background not installed.")
    
    mode = settings.get("mask_remover_mode", "base-nightly")
    # Note: initialization might be heavy
    return Remover(mode=mode, jit=False, device='cuda', ckpt=None)

def process_background_removal(img_input: Image.Image, settings: AppSettings, remover) -> Image.Image:
    """
    核心去背運算邏輯。
    輸入: PIL Image (Any mode)
    輸出: PIL Image (RGBA) - 已經過 Alpha 映射與邊緣處理
    """
    cfg = settings
    
    # Settings
    alpha_threshold = int(cfg.get("mask_default_alpha", 0))
    padding = int(cfg.get("mask_padding", 1)) # Default changed to 1 based on previous file
    blur_radius = int(cfg.get("mask_blur_radius", 3))
    
    # 轉換為 RGBA 並處理全透明像素
    img = img_input.convert('RGBA')
    img_arr = np.array(img)
    
    # (1) 原始 Alpha 與 修正 (全透明填白避免 AI 誤判)
    alpha_orig = img_arr[:, :, 3]
    img_to_ai = img.convert('RGB')
    if np.any(alpha_orig == 0):
        bg_w = Image.new("RGB", img.size, (255, 255, 255))
        bg_w.paste(img_to_ai, mask=img.split()[3])
        img_to_ai = bg_w

    # (2) 生成遮罩 (使用 map 模式)
    is_reverse = bool(cfg.get("mask_reverse", False))
    
    # Remover.process 回傳 PIL.Image (L 模式 if type='map')
    mask_img_ai = remover.process(img_to_ai, type='map', reverse=is_reverse)
    mask_arr_ai = np.array(mask_img_ai.convert('L')) # 0-255

    # (3) Alpha 重新映射 (區間映射)
    min_r = float(cfg.get("mask_batch_min_foreground_ratio", 0.3))
    max_r = float(cfg.get("mask_batch_max_foreground_ratio", 0.8))
    
    curr_ratio = np.mean(mask_arr_ai) / 255.0
    
    if curr_ratio < min_r:
        # 擴張/增強
        gamma = curr_ratio / min_r if min_r > 0 else 1
        mask_arr_ai = (np.power(mask_arr_ai / 255.0, gamma) * 255.0).astype(np.uint8)
    elif curr_ratio > max_r:
        # 縮減/減弱
        gamma = curr_ratio / max_r if max_r > 0 else 1
        mask_arr_ai = (np.power(mask_arr_ai / 255.0, gamma) * 255.0).astype(np.uint8)
    
    # 結合原始 Alpha
    combined_alpha = np.minimum(mask_arr_ai, alpha_orig)

    # (4) Padding -> Blur
    mask_processed = Image.fromarray(combined_alpha)
    if padding > 0:
        mask_processed = mask_processed.filter(ImageFilter.MinFilter(padding * 2 + 1))
    if blur_radius > 0:
        mask_processed = mask_processed.filter(ImageFilter.GaussianBlur(blur_radius))
    
    alpha_final = np.array(mask_processed).astype(np.float32)
    if alpha_threshold > 0:
        alpha_final = np.maximum(alpha_final, alpha_threshold)
    alpha_final = np.clip(alpha_final, 0, 255).astype(np.uint8)

    # 組合最終圖像
    img.putalpha(Image.fromarray(alpha_final, mode="L"))
    return img

# ==========================================
#  OCR Text Detection
# ==========================================

def detect_text_boxes(img_input: Image.Image, settings: AppSettings) -> list:
    """
    使用 OCR 偵測文字區域。
    回傳: List[Tuple(x1, y1, x2, y2)]
    """
    if detect_text_with_ocr is None:
        return []
    
    if not bool(settings.get("mask_batch_detect_text_enabled", True)):
        return []

    try:
        heat = float(settings.get("mask_ocr_heat_threshold", 0.2))
        box = float(settings.get("mask_ocr_box_threshold", 0.6))
        unclip = float(settings.get("mask_ocr_unclip_ratio", 2.3))
        max_c = int(settings.get("mask_ocr_max_candidates", 300))
        
        # 預處理：將影像轉為 RGB 並將透明區域填白
        # 這裡不修改原始 img_input，而是建立副本
        if img_input.mode == 'RGBA':
            background = Image.new("RGB", img_input.size, (255, 255, 255))
            background.paste(img_input, mask=img_input.split()[3])
            ocr_input = background
        else:
            ocr_input = img_input.convert("RGB")

        try:
            results = detect_text_with_ocr(
                ocr_input,
                max_candidates=max_c,
                heat_threshold=heat,
                box_threshold=box,
                unclip_ratio=unclip
            )
        except TypeError:
            # Fallback for older versions
            results = detect_text_with_ocr(
                ocr_input,
                heat_threshold=heat,
                box_threshold=box,
                unclip_ratio=unclip
            )

        boxes = []
        for item in results or []:
            if not item:
                continue
            b = item[0]
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append(tuple(b))
        return boxes

    except Exception as e:
        print(f"[OCR Service] detection failed: {e}")
        return []
