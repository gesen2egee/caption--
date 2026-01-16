
from typing import Tuple, Any
from openai import OpenAI
from .base import BaseProcessor
from ..data import ImageContext, AppSettings
from ..services.llm import prepare_image_for_llm, generate_caption
from ..services.common import unload_all_models

class LLMProcessor(BaseProcessor):
    def __init__(self, settings: AppSettings, override_user_prompt: str = None, is_batch: bool = True):
        super().__init__(settings)
        self.client = None
        self.override_user_prompt = override_user_prompt
        self.is_batch = is_batch

    def prepare(self):
        """Batch 開始前建立連線"""
        api_key = self.settings.llm_api_key
        base_url = self.settings.llm_base_url
        if not api_key:
            raise ValueError("LLM API Key is missing.")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def process(self, ctx: ImageContext) -> Tuple[bool, Any]:
        if not self.client:
            raise RuntimeError("Client not initialized. Call prepare() first.")

        # 1. 取得 Tags Context
        # 若 Sidecar 沒資料，這裡暫時給空字串
        tags = ctx.sidecar.get("tagger_tags", "")
        
        # 2. 檢查 NSFW 跳過 (僅在 Batch 模式下)
        if self.is_batch and bool(self.settings.get("llm_skip_nsfw_on_batch", False)):
            t_lower = tags.lower()
            if "rating:explicit" in t_lower or "rating:questionable" in t_lower:
                # Skip
                return False, None

        # 3. 準備圖片與提示
        img = prepare_image_for_llm(ctx, self.settings)
        sys_prompt = self.settings.system_prompt
        user_tmpl = self.override_user_prompt if self.override_user_prompt is not None else self.settings.user_prompt_template
        model = self.settings.llm_model

        # 4. 執行生成
        content = generate_caption(
            client=self.client,
            model_name=model,
            system_prompt=sys_prompt,
            user_prompt_template=user_tmpl,
            image=img,
            tags_context=tags,
            is_gray_masked=self.settings.get("llm_use_gray_mask", True)
        )

        # 清理 LLM 輸出（移除提示標記）
        from lib.utils import clean_llm_output
        cleaned_content = clean_llm_output(content)

        # 5. 更新 Sidecar
        ctx.sidecar["llm_response"] = cleaned_content
        
        # 将结果也追加到 nl_pages (历史记录)，确保 Batch 模式下也能被后续逻辑 (如 to_txt) 读取
        nl_pages = ctx.sidecar.get("nl_pages", [])
        if not isinstance(nl_pages, list):
            nl_pages = []
        nl_pages.append(cleaned_content)
        ctx.sidecar["nl_pages"] = nl_pages
        
        ctx.save_sidecar()

        # 回傳內容，讓 UI 可以顯示或寫入 txt
        return True, cleaned_content

    def cleanup(self):
        self.client = None
        unload_all_models()
