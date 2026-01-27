# -*- coding: utf-8 -*-
"""
VLM OpenRouter API Worker

使用 OpenRouter API 的視覺語言模型 (VLM)。
支援的模型: mistral_large_2512 等
"""
import base64
import traceback
from io import BytesIO
from typing import Optional, Dict

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.core.dataclasses import ImageData


# 模型預設設定
MODEL_PRESETS = {
    "mistral_large_2512": {
        "model_name": "mistralai/mistral-large-2512",
        "max_tokens": 4096,
    },
    "gpt4o": {
        "model_name": "openai/gpt-4o",
        "max_tokens": 4096,
    },
    "claude_sonnet": {
        "model_name": "anthropic/claude-3.5-sonnet",
        "max_tokens": 4096,
    },
}


class VLMOpenRouterAPIWorker(BaseWorker):
    """
    VLM OpenRouter API Worker
    
    使用 OpenRouter API 進行視覺語言模型推論。
    """
    
    category = "LLM"
    display_name = "OpenRouter / OpenAI API"
    description = "Use remote VLM APIs compatible with OpenAI format"
    default_config = {
         "base_url": "https://openrouter.ai/api/v1",
         "model_name": "mistralai/mistral-large-2512"
    }
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取設定
        self.base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        self.api_key = self.config.get("api_key", "")
        self.model_name = self.config.get("model_name", "mistralai/mistral-large-2512")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.max_image_dim = self.config.get("max_image_dim", 1024)
        self.use_gray_mask = self.config.get("use_gray_mask", True)
    
    @property
    def name(self) -> str:
        return "vlm_openrouter_api"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import openai
            return True
        except ImportError:
            return False
    
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'VLMOpenRouterAPIWorker':
        """從預設設定建立 Worker"""
        preset = MODEL_PRESETS.get(preset_name, MODEL_PRESETS["mistral_large_2512"])
        config = {**preset, **kwargs}
        return cls(config)
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """將圖片編碼為 base64"""
        try:
            img = Image.open(image_path)
            
            # 處理透明通道
            if img.mode == "RGBA" and self.use_gray_mask:
                # 將透明區域填充為灰色
                bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg.convert("RGB")
            elif img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            # 縮放圖片
            if max(img.size) > self.max_image_dim:
                ratio = self.max_image_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # 編碼為 base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        except Exception as e:
            print(f"[VLM] 圖片編碼失敗: {e}")
            return None
    
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行 VLM 推論"""
        try:
            # 驗證輸入
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            
            image_data = input_data.image
            settings = input_data.settings
            
            # 取得提示詞
            system_prompt = ""
            user_prompt = ""
            
            if settings:
                system_prompt = settings.llm_system_prompt or ""
                user_prompt = settings.llm_user_prompt_template or ""
            
            # 從 extra 取得覆蓋值
            system_prompt = input_data.extra.get("system_prompt", system_prompt)
            user_prompt = input_data.extra.get("user_prompt", user_prompt)
            
            # 替換模板變數
            tags_context = image_data.tagger_tags or ""
            user_prompt = user_prompt.replace("{tags}", tags_context)
            
            # 編碼圖片
            image_b64 = self._encode_image(image_data.path)
            if not image_b64:
                return WorkerOutput(success=False, error="圖片編碼失敗")
            
            # 呼叫 API
            from openai import OpenAI
            
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                                    
            # 2. 構建 User Content (支援重複輸入)
            repeat_count = getattr(settings, 'llm_input_repeat_count', 2) if settings else 2
            # Allow override from extra/input
            repeat_count = int(input_data.extra.get("llm_input_repeat_count", repeat_count))
            
            user_content = []
            
            # 基本單元: [User Prompt, Image]
            base_unit = [
                {
                    "type": "text", 
                    "text": f"{user_prompt}"
                }, 
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
            
            for i in range(repeat_count):
                # 在重複的單元之間插入 System Prompt (如果 User 要求的中間插入)
                # Client example: User, Image, System, User, Image
                if i > 0 and system_prompt:
                    user_content.append({
                        "type": "text",
                        "text": f"{system_prompt}"
                    })
                
                user_content.extend(base_unit)

            messages.append({
                "role": "user",
                "content": user_content
            })

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
            )
                        
            result_text = response.choices[0].message.content or ""
            
            # 更新 ImageData
            image_data.llm_result = result_text
            if result_text not in image_data.nl_pages:
                image_data.nl_pages.append(result_text)
            
            return WorkerOutput(
                success=True,
                image=image_data,
                result_text=result_text,
            )
            
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        if not self.api_key:
            return "缺少 API Key"
        return None
