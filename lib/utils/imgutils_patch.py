# -*- coding: utf-8 -*-
"""
Imgutils TIMM Model Patch
支援含有子資料夾的 Repo ID (例如 Makki2104/animetimm/eva02_large_patch14_448.dbv4-full)
"""
import os
import json
import io
import warnings
from threading import Lock
from typing import Optional, Literal, Dict, Any, Union

import pandas as pd
from hbutils.design import SingletonMark
from hfutils.repository import hf_hub_repo_url
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# 從原有的 imgutils 匯入需要的工具
from imgutils.data import ImageTyping, load_image
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, vreplace, ts_lru_cache

FMT_UNSET = SingletonMark('FMT_UNSET')

def split_repo_id(repo_id: str):
    """
    將 repo_id 切分為 (base_repo, subfolder)
    例如: "user/repo/folder" -> ("user/repo", "folder")
    """
    parts = repo_id.split('/')
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return repo_id, None

class PatchedMultiLabelTIMMModel:
    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        self.full_repo_id = repo_id
        self.repo_id, self.subfolder = split_repo_id(repo_id)
        self._model = None
        self._df_tags = None
        self._preprocess = None
        self._default_category_thresholds = None
        self._hf_token = hf_token
        self._lock = Lock()
        self._category_names = {}
        self._name_to_categories = None

    def _get_hf_token(self) -> Optional[str]:
        return self._hf_token or os.environ.get('HF_TOKEN')

    def _get_filename(self, filename: str) -> str:
        if self.subfolder:
            return f"{self.subfolder}/{filename}"
        return filename

    def _open_model(self):
        with self._lock:
            if self._model is None:
                self._model = open_onnx_model(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=self._get_filename('model.onnx'),
                    token=self._get_hf_token(),
                ))
        return self._model

    def _open_tags(self):
        with self._lock:
            if self._df_tags is None:
                self._df_tags = pd.read_csv(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=self._get_filename('selected_tags.csv'),
                    token=self._get_hf_token(),
                ), keep_default_na=False)

                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=self._get_filename('categories.json'),
                        token=self._get_hf_token(),
                ), 'r') as f:
                    d_category_names = {cate_item['category']: cate_item['name'] for cate_item in json.load(f)}
                    self._name_to_categories = {}
                    for category in sorted(set(self._df_tags['category'])):
                        self._category_names[category] = d_category_names[category]
                        self._name_to_categories[self._category_names[category]] = category
        return self._df_tags

    def _open_preprocess(self):
        with self._lock:
            if self._preprocess is None:
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=self._get_filename('preprocess.json')
                ), 'r') as f:
                    data_ = json.load(f)
                    from imgutils.preprocess import create_pillow_transforms
                    test_trans = create_pillow_transforms(data_['test'])
                    val_trans = create_pillow_transforms(data_['val'])
                    self._preprocess = val_trans, test_trans
        return self._preprocess

    def _open_default_category_thresholds(self):
        with self._lock:
            if self._default_category_thresholds is None:
                try:
                    df_category_thresholds = pd.read_csv(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=self._get_filename('thresholds.csv')
                    ), keep_default_na=False)
                except (EntryNotFoundError,):
                    self._default_category_thresholds = {}
                else:
                    self._default_category_thresholds = {}
                    for item in df_category_thresholds.to_dict('records'):
                        if item['category'] not in self._default_category_thresholds:
                            self._default_category_thresholds[item['category']] = item['threshold']
        return self._default_category_thresholds

    def _raw_predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test'):
        image = load_image(image, force_background='white', mode='RGB')
        model = self._open_model()
        val_trans, test_trans = self._open_preprocess()
        if preprocessor == 'test':
            trans = test_trans
        elif preprocessor == 'val':
            trans = val_trans
        else:
            raise ValueError(f'Unknown processor, "test" or "val" expected but {preprocessor!r} found.')

        input_ = trans(image)[None, ...]
        output_names = [output.name for output in model.get_outputs()]
        output_values = model.run(output_names, {'input': input_})
        return {name: value[0] for name, value in zip(output_names, output_values)}

    def predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test',
                thresholds: Union[float, Dict[Any, float]] = None, use_tag_thresholds: bool = True,
                fmt=FMT_UNSET):
        df_tags = self._open_tags()
        values = self._raw_predict(image, preprocessor=preprocessor)
        prediction = values['prediction']
        tags = {}

        if fmt is FMT_UNSET:
            fmt = tuple(self._category_names[category] for category in sorted(set(df_tags['category'].tolist())))

        default_category_thresholds = self._open_default_category_thresholds()
        if 'best_threshold' in self._df_tags:
            default_tag_thresholds = self._df_tags['best_threshold']
        else:
            default_tag_thresholds = None
            
        for category in sorted(set(df_tags['category'].tolist())):
            mask = df_tags['category'] == category
            tag_names = df_tags['name'][mask]
            category_pred = prediction[mask]

            if isinstance(thresholds, float):
                category_threshold = thresholds
            elif isinstance(thresholds, dict) and \
                    (category in thresholds or self._category_names[category] in thresholds):
                if category in thresholds:
                    category_threshold = thresholds[category]
                elif self._category_names[category] in thresholds:
                    category_threshold = thresholds[self._category_names[category]]
                else:
                    assert False
            elif use_tag_thresholds and default_tag_thresholds is not None:
                category_threshold = default_tag_thresholds[mask]
            else:
                if use_tag_thresholds:
                    warnings.warn(f'Tag thresholds not supported in model {self.full_repo_id!r}.')
                if category in default_category_thresholds:
                    category_threshold = default_category_thresholds[category]
                else:
                    category_threshold = 0.4

            mask = category_pred >= category_threshold
            tag_names = tag_names[mask].tolist()
            category_pred = category_pred[mask].tolist()
            cate_tags = dict(sorted(zip(tag_names, category_pred), key=lambda x: (-x[1], x[0])))
            values[self._category_names[category]] = cate_tags
            tags.update(cate_tags)

        values['tag'] = tags
        return vreplace(fmt, values)

@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None):
    return PatchedMultiLabelTIMMModel(repo_id=repo_id, hf_token=hf_token)

def multilabel_timm_predict_patched(image: ImageTyping, repo_id: str,
                                    preprocessor: Literal['test', 'val'] = 'test',
                                    thresholds: Union[float, Dict[Any, float]] = None, 
                                    use_tag_thresholds: bool = True,
                                    fmt=FMT_UNSET, hf_token: Optional[str] = None):
    model = _open_models_for_repo_id(repo_id=repo_id, hf_token=hf_token)
    return model.predict(
        image=image,
        preprocessor=preprocessor,
        thresholds=thresholds,
        use_tag_thresholds=use_tag_thresholds,
        fmt=fmt,
    )
