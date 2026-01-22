# -*- coding: utf-8 -*-
"""
Imgutils Generic Tagger Worker (Timm Models)

使用 imgutils.generic.multilabel_timm_predict 進行標籤識別。
支援自定義 Repo ID 的模型 (如 Makki2104/animetimm 系列)。
"""
import traceback
from typing import Optional, Dict

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.core.dataclasses import ImageData


# 模型預設設定
MODEL_PRESETS = {
    "EVA02_Large_Animetimm": {
        "repo_id": "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8,
    },
    "ConvNextV2_Huge_Animetimm": {
        "repo_id": "Makki2104/animetimm/convnextv2_huge.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8,
    },
    "SwinV2_Base_Animetimm": {
        "repo_id": "Makki2104/animetimm/swinv2_base_window8_256.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8,
    },
}


class TaggerImgutilsGenericWorker(BaseWorker):
    """
    Imgutils Generic Tagger Worker
    
    使用 imgutils.generic.multilabel_timm_predict 支援更多模型。
    """
    
    category = "TAGGER"
    display_name = "Local (imgutils-timm)"
    description = "Run generic Timm models via imgutils (e.g. Makki2104/animetimm)"
    default_config = {
        "repo_id": "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8
    }
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取設定
        # 這裡的 tagger_model 會對應到 UI 的設定欄位，我們將其作為 repo_id 使用
        self.repo_id = self.config.get("tagger_model") or self.config.get("repo_id") or "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full"
        self.general_threshold = self.config.get("general_threshold", 0.25)
        self.character_threshold = self.config.get("character_threshold", 0.8)
    
    @property
    def name(self) -> str:
        return "tagger_imgutils_generic"
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            from imgutils.generic import multilabel_timm_predict
            return True
        except ImportError:
            return False

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'TaggerImgutilsGenericWorker':
        """從預設設定建立 Worker"""
        preset = MODEL_PRESETS.get(preset_name, MODEL_PRESETS["EVA02_Large_Animetimm"])
        config = {**preset, **kwargs}
        # 將 repo_id 對應到 tagger_model 以符合 UI 慣例
        if "repo_id" in config:
            config["tagger_model"] = config.pop("repo_id")
        return cls(config)
    
    def _process_convnext_huge(self, image_data: ImageData):
        """
        Special handling for Makki2104/animetimm/convnextv2_huge.dbv4-full
        using torch and timm as requested by user.
        """
        try:
            import torch
            import json
            import pandas as pd
            from huggingface_hub import hf_hub_download
            from timm import create_model
            from imgutils.preprocess import create_torchvision_transforms
            from imgutils.tagging import tags_to_text, remove_underline
            from imgutils.data import load_image
            from lib.utils.imgutils_patch import split_repo_id

            repo_id, subfolder = split_repo_id(self.repo_id)
            
            # 1. Load Preprocessor
            config_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{subfolder}/preprocess.json" if subfolder else "preprocess.json",
                repo_type='model'
            )
            with open(config_path, 'r') as f:
                preprocessor = create_torchvision_transforms(json.load(f)['test'])

            # 2. Load Model
            # Usage pattern from user: create_model(f'hf-hub:{repo_id}', pretrained=True)
            # Since we have a subfolder structure which timm might not natively resolve nicely via hf-hub string,
            # we download manually and load.
            # However, user suggested: model = create_model(f'hf-hub:{repo_id}', pretrained=True)
            # If we assume 'convnextv2_huge' is the architecture:
            
            # Try to find model file
            model_file = "model.safetensors" # common default
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/{model_file}" if subfolder else model_file)
            except:
                model_file = "pytorch_model.bin"
                model_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/{model_file}" if subfolder else model_file)

            model = create_model('convnextv2_huge', pretrained=False, num_classes=12476) # 12476 from user comments
            # Load weights
            # Determine if safetensors or bin
            if model_file.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            
            model.load_state_dict(state_dict)
            
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()

            # 3. Predict
            image = load_image(image_data.path, force_background='white', mode='RGB')
            input_ = preprocessor(image).unsqueeze(0)
            if torch.cuda.is_available():
                input_ = input_.cuda()

            with torch.no_grad():
                output = model(input_)
                prediction = torch.sigmoid(output)[0].cpu()

            # 4. Load Tags
            tags_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{subfolder}/selected_tags.csv" if subfolder else "selected_tags.csv",
                repo_type='model'
            )
            df_tags = pd.read_csv(tags_path, keep_default_na=False)
            tags_names = df_tags['name']
            
            # 5. Process Results
            # Mapping logic
            if 'best_threshold' in df_tags.columns:
                thresholds = df_tags['best_threshold'].values
                mask = prediction.numpy() >= thresholds
            else:
                # Fallback if no per-tag threshold
                mask = prediction.numpy() >= self.general_threshold
            
            # Create dict of {tag: score}
            result_tags = dict(zip(tags_names[mask].tolist(), prediction[mask].tolist()))
            
            # Categorize
            # We need to map back to categories (general, character, etc)
            # User snippet didn't show categorization explicitly, but we need it for our data structure
            # df_tags usually has 'category' column
            
            general = {}
            character = {}
            rating = {} # Model might not output rating? User snippet shows tags like 'sensitive', '1girl'.
            # 'sensitive' usually maps to rating/meta.
            # '1girl' is char/general?
            
            # Standard imgutils categorization:
            # category 0: General
            # category 4: Character
            # category 9: Rating
            # Use 'category' column if exists
            
            processed_tags = {}
            for tag, score in result_tags.items():
                # Filter by threshold based on our settings?
                # The user snippet uses 'best_threshold' from CSV. 
                # Our settings have general_threshold/character_threshold. 
                # We should probably respect settings if they are higher, or just use what we got.
                # Let's filter by our generic thresholds for safety/consistency if needed.
                
                # Check category in df
                row = df_tags[df_tags['name'] == tag]
                if not row.empty:
                    cat_id = row.iloc[0]['category']
                    # 0: general, 4: character, 9: rating
                    if cat_id == 0:
                        if score >= self.general_threshold:
                            general[tag] = float(score)
                    elif cat_id == 4:
                        if score >= self.character_threshold:
                            character[tag] = float(score)
                    elif cat_id == 9:
                        rating[tag] = float(score)
            
            # Normalize
            features = {remove_underline(k): v for k, v in general.items()}
            chars = {remove_underline(k): v for k, v in character.items()}
             
            all_tags = {**features, **chars}
            tags_text = tags_to_text(all_tags)
            
            image_data.tagger_tags = tags_text
            image_data.tagger_rating = rating
            image_data.tagger_features = features
            image_data.tagger_chars = chars
            
            return WorkerOutput(
                success=True,
                image=image_data,
                result_text=tags_text,
                result_data={
                    "rating": rating,
                    "features": features,
                    "chars": chars,
                }
            )

        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))

    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行標籤識別"""
        if "convnextv2_huge" in self.repo_id:
            return self._process_convnext_huge(input_data.image)

        try:
            # 驗證輸入
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            
            image_data = input_data.image
            
            # 載入圖片
            img = Image.open(image_data.path)
            
            # 呼叫 imgutils (使用支援子資料夾的修正版)
            from lib.utils.imgutils_patch import multilabel_timm_predict_patched
            from imgutils.tagging import tags_to_text, remove_underline
            
            # 執行預測
            general, character, rating = multilabel_timm_predict_patched(
                img,
                repo_id=self.repo_id,
                fmt=("general", "character", "rating"),
            )
            
            # 過濾閾值
            general = {k: v for k, v in general.items() if v >= self.general_threshold}
            character = {k: v for k, v in character.items() if v >= self.character_threshold}
            
            # 正規化標籤 (移除底線)
            features = {remove_underline(k): v for k, v in general.items()}
            chars = {remove_underline(k): v for k, v in character.items()}
            
            # 轉換為文字
            all_tags = {**features, **chars}
            tags_text = tags_to_text(all_tags)
            
            # 更新 ImageData
            image_data.tagger_tags = tags_text
            image_data.tagger_rating = rating # rating 是一組分數，如 {'safe': 0.9, ...}
            image_data.tagger_features = features
            image_data.tagger_chars = chars
            
            return WorkerOutput(
                success=True,
                image=image_data,
                result_text=tags_text,
                result_data={
                    "rating": rating,
                    "features": features,
                    "chars": chars,
                }
            )
            
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        return None
