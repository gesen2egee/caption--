# 清理 caption.py 重複代碼的腳本
import sys

def clean_caption_py(input_file, output_file):
    """
    從 caption.py 中刪除重複的類別定義
    保留：
    - 第 1-127 行：導入和環境設定
    - 第 1120-結尾：MainWindow 類別和主程式
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 保留第 1-127 行（索引 0-126）
    header = lines[0:127]
    
    # 保留第 1120 行到結尾（索引 1119-）
    main_window_and_entry = lines[1119:]
    
    # 組合並寫入
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(header)
        f.write('\n')
        f.write('# ==========================================\n')
        f.write('#  Main Window\n')
        f.write('# ==========================================\n')
        f.write('# UI classes (StrokeCanvas, StrokeEraseDialog, TagButton, TagFlowWidget,\n')
        f.write('# AdvancedFindReplaceDialog, SettingsDialog) have been moved to lib.ui\n')
        f.write('\n')
        f.writelines(main_window_and_entry)
    
    print(f"✓ 已清理 {input_file}")
    print(f"✓ 輸出到 {output_file}")
    print(f"✓ 原始行數: {len(lines)}")
    print(f"✓ 新行數: {len(header) + len(main_window_and_entry) + 6}")

if __name__ == '__main__':
    clean_caption_py('e:\\caption--\\caption.py', 'e:\\caption--\\caption_clean.py')
