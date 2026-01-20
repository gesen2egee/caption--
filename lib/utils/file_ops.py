# -*- coding: utf-8 -*-
"""
檔案操作模組

處理圖片備份、還原、NPZ 刪除等檔案操作。
"""
import os
import shutil

from lib.utils.sidecar import load_image_sidecar, save_image_sidecar


def delete_matching_npz(image_path: str) -> int:
    """
    刪除與圖檔名匹配的 npz 檔案。
    例如圖檔 '1b7f4f85fac7f8f7076fa528e95176fb.webp' 
    會匹配 '1b7f4f85fac7f8f7076fa528e95176fb_0849x0849_sdxl.npz'
    回傳刪除的檔案數量。
    """
    if not image_path:
        return 0
    
    try:
        src_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        deleted = 0
        for f in os.listdir(src_dir):
            if f.endswith(".npz") and f.startswith(base_name):
                npz_path = os.path.join(src_dir, f)
                try:
                    os.remove(npz_path)
                    deleted += 1
                    print(f"[NPZ] 已刪除: {f}")
                except Exception as e:
                    print(f"[NPZ] 刪除失敗 {f}: {e}")
        return deleted
    except Exception as e:
        print(f"[NPZ] delete_matching_npz 錯誤: {e}")
        return 0


# ==========================================
#  Raw Image Backup / Restore
# ==========================================

def get_raw_image_dir(image_path: str) -> str:
    """取得 raw_image 備份資料夾路徑"""
    return os.path.join(os.path.dirname(image_path), "raw_image")


def has_raw_backup(image_path: str) -> bool:
    """
    檢查圖片是否已有原圖備份。
    檢查 sidecar JSON 中的 raw_backup_path 欄位。
    """
    sidecar = load_image_sidecar(image_path)
    if "raw_backup_path" not in sidecar and "raw_image_rel_path" not in sidecar:
        return False
    
    rel_path = sidecar.get("raw_backup_path") or sidecar.get("raw_image_rel_path")
    if not rel_path:
        return False
    
    src_dir = os.path.dirname(image_path)
    abs_raw_path = os.path.normpath(os.path.join(src_dir, rel_path))
    return os.path.exists(abs_raw_path)


def backup_original_image(image_path: str) -> bool:
    """
    修改圖片前先備份原圖 (若為首次修改)。
    備份到同層級的 raw_image 資料夾。
    並在 sidecar 記錄 'raw_image_rel_path'。
    """
    try:
        sidecar = load_image_sidecar(image_path)
        
        # 若已經有備份紀錄，先檢查檔案是否存在
        if "raw_image_rel_path" in sidecar:
            rel_path = sidecar["raw_image_rel_path"]
            src_dir = os.path.dirname(image_path)
            abs_raw_path = os.path.normpath(os.path.join(src_dir, rel_path))
            if os.path.exists(abs_raw_path):
                return True

        # 執行備份
        src_dir = os.path.dirname(image_path)
        raw_dir = os.path.join(src_dir, "raw_image")
        os.makedirs(raw_dir, exist_ok=True)
        
        fname = os.path.basename(image_path)
        dest_path = os.path.join(raw_dir, fname)
        
        if not os.path.exists(dest_path):
            shutil.copy2(image_path, dest_path)
        
        rel_path = os.path.relpath(dest_path, src_dir)
        sidecar["raw_image_rel_path"] = rel_path
        save_image_sidecar(image_path, sidecar)
        return True
    except Exception as e:
        print(f"[Backup] 備份失敗 {image_path}: {e}")
        return False


def restore_original_image(image_path: str) -> bool:
    """
    嘗試從 sidecar 記錄的 raw_image 還原圖片。
    """
    try:
        sidecar = load_image_sidecar(image_path)
        if "raw_image_rel_path" not in sidecar:
            return False
            
        rel_path = sidecar["raw_image_rel_path"]
        src_dir = os.path.dirname(image_path)
        abs_raw_path = os.path.normpath(os.path.join(src_dir, rel_path))
        
        if os.path.exists(abs_raw_path):
            shutil.copy2(abs_raw_path, image_path)
            if "masked_text" in sidecar: 
                del sidecar["masked_text"]
            if "masked_background" in sidecar: 
                del sidecar["masked_background"]
            save_image_sidecar(image_path, sidecar)
            return True
        return False
    except Exception as e:
        print(f"[Restore] 還原失敗 {image_path}: {e}")
        return False


def backup_raw_image(image_path: str) -> bool:
    """
    備份原圖到 raw_image 資料夾。
    - 如果已有備份，不重複備份
    - 備份後在 sidecar JSON 中記錄相對路徑
    - 回傳 True 表示有執行備份，False 表示已存在備份
    """
    if has_raw_backup(image_path):
        return False
    return backup_original_image(image_path)


def restore_raw_image(image_path: str) -> bool:
    """
    從 raw_image 還原原圖。
    - 如果沒有備份紀錄，回傳 False
    - 還原後清除 sidecar 中的 mask 標記
    - 回傳 True 表示還原成功
    """
    return restore_original_image(image_path)


def delete_raw_backup(image_path: str) -> bool:
    """
    刪除原圖備份（當使用者確認不需要還原時）。
    - 刪除 raw_image 中的備份檔案
    - 清除 sidecar 中的備份路徑
    """
    try:
        sidecar = load_image_sidecar(image_path)
        rel_path = sidecar.get("raw_backup_path") or sidecar.get("raw_image_rel_path")
        
        if not rel_path:
            return False
        
        src_dir = os.path.dirname(image_path)
        abs_raw_path = os.path.normpath(os.path.join(src_dir, rel_path))
        
        if os.path.exists(abs_raw_path):
            os.remove(abs_raw_path)
        
        if "raw_backup_path" in sidecar:
            del sidecar["raw_backup_path"]
        if "raw_image_rel_path" in sidecar:
            del sidecar["raw_image_rel_path"]
        
        save_image_sidecar(image_path, sidecar)
        return True
    except Exception as e:
        print(f"[DeleteBackup] 刪除備份失敗 {image_path}: {e}")
        return False
