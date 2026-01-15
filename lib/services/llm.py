
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional

from ..data import AppSettings, ImageContext
from ..utils import load_image_sidecar

def prepare_image_for_llm(ctx: ImageContext, settings: AppSettings) -> Image.Image:
    """
    準備傳送給 LLM 的圖片。
    包含：
    1. 決定要讀哪張圖 (原圖、Backup 或 Unmask)
    2. 處理透明度 (Gray Mask)
    3. 縮放至最大限制
    """
    use_gray = bool(settings.get("llm_use_gray_mask", True))
    max_dim = int(settings.get("llm_max_image_dimension", 1024))
    
    img_path_to_open = ctx.path

    # 若不使用灰底，嘗試尋找 unmask 資料夾中的去背前原檔
    # (這是原本 caption.py 的特定邏輯)
    if not use_gray:
        src_dir = os.path.dirname(ctx.path)
        stem = os.path.splitext(os.path.basename(ctx.path))[0]
        unmask_dir = os.path.join(src_dir, "unmask")
        if os.path.exists(unmask_dir):
            for f in os.listdir(unmask_dir):
                if os.path.splitext(f)[0] == stem:
                    img_path_to_open = os.path.join(unmask_dir, f)
                    break
    
    # 開啟圖片 (建立副本以免影響快取)
    try:
        if img_path_to_open == ctx.path:
            img = ctx.get_image(mode='RGBA').copy()
        else:
            img = Image.open(img_path_to_open).convert('RGBA')
    except Exception:
        # Fallback
        img = Image.open(ctx.path).convert('RGBA')

    # 處理 Mask (灰底)
    sidecar = ctx.sidecar
    is_masked_in_app = sidecar.get("masked_text", False) or sidecar.get("masked_background", False)
    
    has_alpha = False
    if img.mode == 'RGBA':
        has_alpha = True
        if use_gray:
            # 強制變全灰 (無論 alpha 多少，只要有透明度就變灰)
            # 建立灰色底圖 (136, 136, 136)
            canvas = Image.new("RGB", img.size, (136, 136, 136))
            alpha = img.getchannel('A')
            # Binary: alpha == 255 -> keep pixel (255), else -> 0 (use gray bg)
            # 注意: 原邏輯是 alpha < 255 則視為透明
            mask = alpha.point(lambda p: 255 if p == 255 else 0)
            canvas.paste(img, mask=mask)
            img = canvas
        else:
            img = img.convert('RGB')
    else:
        img = img.convert('RGB')

    # Resize
    target_size = max_dim
    if img.width > target_size or img.height > target_size:
        ratio = min(target_size / img.width, target_size / img.height)
        if ratio < 1:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    return img

def encode_image_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_caption(
    client, 
    model_name: str, 
    system_prompt: str, 
    user_prompt_template: str, 
    image: Image.Image, 
    tags_context: str,
    is_gray_masked: bool = False
) -> str:
    """
    呼叫 OpenAI Compatible API 產生描述。
    """
    img_b64 = encode_image_base64(image)
    img_url = f"data:image/jpeg;base64,{img_b64}"

    # 決定提示詞前綴
    prompt_prefix = ""
    if is_gray_masked:
        prompt_prefix = "這是一張經過去背處理的圖像，背景已填滿灰色，請忽視灰色區域並針對主體進行描述。\n"
    
    # 組合 User Prompt
    # 支援 {LLM處理結果} 和 {tags} 佔位符
    final_user_content = prompt_prefix + user_prompt_template.replace("{LLM處理結果}", tags_context)
    final_user_content = final_user_content.replace("{tags}", tags_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": final_user_content},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        raise e
