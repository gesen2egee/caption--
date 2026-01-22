# -*- coding: utf-8 -*-
"""
Worker Registry

負責管理與自動發現所有的 Worker。
"""
import inspect
import pkgutil
import importlib
from typing import Dict, List, Type, Optional
from lib.workers.base import BaseWorker

class WorkerRegistry:
    _instance = None
    _workers: Dict[str, Dict[str, Type[BaseWorker]]] = {
        "TAGGER": {},
        "LLM": {},
        "UNMASK": {},
        "MASK_TEXT": {},
        "RESTORE": {},
        "OTHER": {}
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WorkerRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, worker_cls: Type[BaseWorker]):
        """註冊一個 Worker 類別"""
        if not issubclass(worker_cls, BaseWorker):
            return
        
        # 實例化一次以取得屬性 (或是改成 classmethod/property)
        # 這裡假設 category 和 name 是類別屬性或 property
        try:
            # 由於 BaseWorker 定義為 property，我們需要實例化或改變設計。
            # 為了避免副作用，建議將 category/name/display_name 改為 class property。
            # 但為了相容性，我們先暫時實例化一個不帶參數的 (如果 __init__ 允許 None)
            # 或者我們修改 BaseWorker 讓他變成 classmethod
            
            # 這裡我們先假設 Worker 已經被修改為提供 class level metadata
            # 如果沒有，我們可能需要調整策略。
            # 這邊採用: 先修改 BaseWorker 結構，再回來完善 register 邏輯。
            pass
        except Exception:
            pass

    @classmethod
    def add_worker(cls, category: str, name: str, worker_cls: Type[BaseWorker]):
        if category not in cls._workers:
            cls._workers[category] = {}
        cls._workers[category][name] = worker_cls

    @classmethod
    def get_worker_class(cls, category: str, name: str) -> Optional[Type[BaseWorker]]:
        return cls._workers.get(category, {}).get(name)

    @classmethod
    def get_workers(cls, category: str) -> List[Dict]:
        """回傳該分類下所有 Worker 的元數據 (用於 UI 列表)"""
        results = []
        for name, worker_cls in cls._workers.get(category, {}).items():
            results.append({
                "name": name,
                "display_name": getattr(worker_cls, "display_name", name),
            })
        return results

    @classmethod
    def has_available_workers(cls, category: str) -> bool:
        """檢查該分類下是否有可用的 Worker"""
        return len(cls._workers.get(category, {})) > 0

    @classmethod
    def scan_workers(cls):
        """掃描 lib.workers 下的所有模組並自動註冊"""
        import lib.workers as workers_pkg
        
        path = workers_pkg.__path__
        prefix = workers_pkg.__name__ + "."

        for _, name, _ in pkgutil.iter_modules(path, prefix):
            try:
                module = importlib.import_module(name)
                for item_name, item in inspect.getmembers(module):
                    if (inspect.isclass(item) and 
                        issubclass(item, BaseWorker) and 
                        item is not BaseWorker):
                        
                        # 檢查可用性
                        if not item.is_available():
                            continue

                        # 取得 Metadata
                        # 這裡假設我們在子類別定義了 class attribute
                        category = getattr(item, "category", "OTHER")
                        
                        # 修正: name 可能是 property，若是則需實例化才能取得字串值
                        worker_name = getattr(item, "name", None)
                        if isinstance(worker_name, property) or worker_name is None:
                            try:
                                # 嘗試實例化 (假設 __init__ 支援不帶參數或 default=None)
                                instance = item()
                                worker_name = instance.name
                            except Exception:
                                print(f"[WorkerRegistry] Failed to instantiate {item_name} to get name")
                                worker_name = None

                        if worker_name and isinstance(worker_name, str):
                             cls.add_worker(category, worker_name, item)
            except Exception as e:
                print(f"[WorkerRegistry] Failed to import {name}: {e}")

# Global instance helper
_registry = WorkerRegistry()

def scan_workers():
    _registry.scan_workers()

def get_registry():
    return _registry
