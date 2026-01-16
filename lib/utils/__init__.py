
import os
import json
import re
import shutil
import base64
import fnmatch
import csv
from io import BytesIO
from urllib.request import urlopen, Request
from pathlib import Path
from PIL import Image

# Import constants if needed
try:
    from .const import TAGS_CSV_URL_RAW, TAGS_CSV_LOCAL
except ImportError:
    TAGS_CSV_URL_RAW = "https://raw.githubusercontent.com/waldolin/a1111-sd-webui-tagcomplete-TW/main/tags/Tags-tw-full-pack.csv"
    TAGS_CSV_LOCAL = "Tags.csv"

# ==========================================
#  Helpers: Image / Checkerboard
# ==========================================

def create_checkerboard_png_bytes():
    """
    生成 16x16 棋盤格 PNG bytes（不走 data URI，避免 Qt stylesheet pixmap 警告）
    """
    try:
        w, h = 16, 16
        img = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        pixels = img.load()
        color = (220, 220, 220, 255)

        for y in range(h):
            for x in range(w):
                if (x < 8 and y < 8) or (x >= 8 and y >= 8):
                    pixels[x, y] = color

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()
    except Exception as e:
        print(f"Error creating checkerboard: {e}")
        # 1x1 透明 PNG
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

# ==========================================
#  Helpers: File I/O & Sidecar
# ==========================================

def delete_matching_npz(image_path: str) -> int:
    """
    刪除與圖檔名匹配的 npz 檔案。
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
                except Exception:
                    pass
        return deleted
    except Exception as e:
        print(f"[NPZ] delete_matching_npz 錯誤: {e}")
        return 0


def image_sidecar_json_path(image_path: str) -> str:
    """取得圖片對應的 sidecar JSON 路徑"""
    return os.path.splitext(image_path)[0] + ".json"


def load_image_sidecar(image_path: str) -> dict:
    """
    載入圖片對應的 sidecar JSON。
    """
    p = image_sidecar_json_path(image_path)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[Sidecar] 載入失敗 {p}: {e}")
    return {}


def save_image_sidecar(image_path: str, data: dict):
    """儲存圖片對應的 sidecar JSON"""
    p = image_sidecar_json_path(image_path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Sidecar] 儲存失敗 {p}: {e}")

# ==========================================
#  Helpers: Backup / Restore
# ==========================================

def get_raw_image_dir(image_path: str) -> str:
    return os.path.join(os.path.dirname(image_path), "raw_image")

def has_raw_backup(image_path: str) -> bool:
    """檢查是否已經有備份原圖"""
    sidecar = load_image_sidecar(image_path)
    raw_rel = sidecar.get("raw_backup_path", "")
    if not raw_rel:
        return False
    
    src_dir = os.path.dirname(image_path)
    raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
    return os.path.exists(raw_abs)

def backup_raw_image(image_path: str) -> bool:
    """
    備份原圖到 raw_image 資料夾。
    - 如果已有備份，不重複備份
    - 備份後在 sidecar JSON 中記錄相對路徑
    """
    if not image_path or not os.path.exists(image_path):
        return False
    
    # 檢查是否已有備份檢查
    if has_raw_backup(image_path):
        return False
    
    try:
        src_dir = os.path.dirname(image_path)
        raw_dir = get_raw_image_dir(image_path)
        os.makedirs(raw_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        dest_path = os.path.join(raw_dir, filename)
        
        # 避免檔名衝突
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            for i in range(1, 9999):
                dest_path = os.path.join(raw_dir, f"{base}_{i}{ext}")
                if not os.path.exists(dest_path):
                    break
        
        # 複製原檔
        shutil.copy2(image_path, dest_path)
        
        # 計算相對路徑並儲存到 sidecar
        rel_path = os.path.relpath(dest_path, src_dir)
        sidecar = load_image_sidecar(image_path)
        sidecar["raw_backup_path"] = rel_path
        save_image_sidecar(image_path, sidecar)
        
        return True
    except Exception as e:
        print(f"[Backup] 備份失敗 {image_path}: {e}")
        return False

def restore_raw_image(image_path: str) -> bool:
    """
    從 raw_image 還原原圖。
    """
    if not image_path:
        return False
    
    sidecar = load_image_sidecar(image_path)
    raw_rel = sidecar.get("raw_backup_path", "")
    
    if not raw_rel:
        return False
    
    src_dir = os.path.dirname(image_path)
    raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
    
    if not os.path.exists(raw_abs):
        return False
    
    try:
        # 複製備份回原位置
        shutil.copy2(raw_abs, image_path)
        
        # 清除 sidecar 中的 mask 標記，但保留備份路徑 (因隨時可能再還原)
        sidecar["masked_background"] = False
        sidecar["masked_text"] = False
        save_image_sidecar(image_path, sidecar)
        return True
    except Exception as e:
        print(f"[Restore] 還原失敗 {image_path}: {e}")
        return False

def delete_raw_backup(image_path: str) -> bool:
    """刪除原圖備份"""
    if not image_path:
        return False
    
    sidecar = load_image_sidecar(image_path)
    raw_rel = sidecar.get("raw_backup_path", "")
    
    if not raw_rel:
        return False
    
    src_dir = os.path.dirname(image_path)
    raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
    
    try:
        if os.path.exists(raw_abs):
            os.remove(raw_abs)
        
        if "raw_backup_path" in sidecar:
            del sidecar["raw_backup_path"]
        save_image_sidecar(image_path, sidecar)
        return True
    except Exception as e:
        print(f"[Backup] 刪除備份失敗 {image_path}: {e}")
        return False

# ==========================================
#  Helpers: String Processing
# ==========================================

def extract_bracket_content(text):
    return re.findall(r'\{(.*?)\}', text)


def smart_parse_tags(text):
    """
    Parses text into a list of dictionaries {'text': str, 'trans': str}.
    """
    if not text:
        return []

    clean_text = text.strip()
    if not clean_text:
        return []

    parsed_items = []
    lines = [l.strip() for l in clean_text.split('\n') if l.strip()]

    is_sentence_mode = False
    if len(lines) > 1:
        for line in lines:
            if (line.startswith("(") and line.endswith(")")) or \
               (line.startswith("（") and line.endswith("）")):
                is_sentence_mode = True
                break
    elif "." in clean_text and "," not in clean_text and len(clean_text) > 50:
        is_sentence_mode = True
        lines = [l.strip() for l in clean_text.replace(". ", ".\n").split('\n') if l.strip()]

    if is_sentence_mode:
        i = 0
        while i < len(lines):
            current_line = lines[i]
            if (current_line.startswith("(") and current_line.endswith(")")) or \
               (current_line.startswith("（") and current_line.endswith("）")):
                i += 1
                continue

            trans = None
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if (next_line.startswith("(") and next_line.endswith(")")) or \
                   (next_line.startswith("（") and next_line.endswith("）")):
                    trans = next_line[1:-1].strip()
                    i += 1

            parsed_items.append({'text': current_line, 'trans': trans})
            i += 1

    else:
        segments = clean_text.replace("\n", ",").split(",")
        for s in segments:
            if s.strip():
                parsed_items.append({'text': s.strip(), 'trans': None})

    return parsed_items


def is_basic_character_tag(text: str, cfg: dict) -> bool:
    """
    判定一段文字（tag 或句子）是否為特徵內容。
    """
    if not text:
        return False
    
    t = text.strip().lower()
    
    bl_words = [w.strip().lower() for w in cfg.get("char_tag_blacklist_words", []) if w.strip()]
    wl_words = [w.strip().lower() for w in cfg.get("char_tag_whitelist_words", []) if w.strip()]
    
    if not bl_words:
        return False
    
    has_blacklist = any(bw in t for bw in bl_words)
    if not has_blacklist:
        return False
    
    has_whitelist = any(ww in t for ww in wl_words)
    if has_whitelist:
        return False
    
    return True


def normalize_for_match(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace(", ", "").replace(",", "")
    t = t.strip()
    t = t.rstrip(".")
    return t.strip()


def cleanup_csv_like_text(text: str, force_lower: bool = False) -> str:
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    result = ", ".join(parts)
    if force_lower:
        result = result.lower()
    return result


def split_csv_like_text(text: str):
    return [p.strip() for p in text.split(",") if p.strip()]

def remove_underline(text: str) -> str:
    if not text:
        return ""
    return text.replace("_", " ").strip()

def try_tags_to_text_list(tags_obj, remove_underline_func=None):
    if not tags_obj: return []
    if isinstance(tags_obj, dict):
        return list(tags_obj.keys())
    return []

# ==========================================
#  Helpers: CSV / Translation
# ==========================================

def load_translations(csv_path=TAGS_CSV_LOCAL):
    """
    載入標籤翻譯 CSV 文件 (本地)
    
    注意：不再從網路下載，請確保 Tags.csv 已存在於專案根目錄
    """
    translations = {}
    
    if not os.path.exists(csv_path):
        print(f"[Tags.csv] 本地檔案不存在: {csv_path}")
        return translations

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip().replace("_", " ")
                    translations[key] = row[1].strip()
    except Exception as e:
        print(f"[Tags.csv] 載入失敗: {e}")
        
    return translations


# ==========================================
#  Danbooru Filter
# ==========================================

class DanbooruQueryFilter:
    """
    Danbooru-style query parser and matcher.
    Supports: AND (space), OR, NOT (-), grouping (()), wildcards (*), rating shortcuts, order.
    """

    def __init__(self, query: str):
        self.query = query.strip()
        self.order_mode = None  # 'landscape' or 'portrait'
        self._parse_order()

    def _parse_order(self):
        match = re.search(r'\border:(landscape|portrait)\b', self.query, re.IGNORECASE)
        if match:
            self.order_mode = match.group(1).lower()
            self.query = re.sub(r'\border:(landscape|portrait)\b', '', self.query, flags=re.IGNORECASE).strip()

    def _normalize(self, text: str) -> str:
        return text.lower().replace("_", " ").strip()

    def _expand_rating(self, term: str) -> str:
        rating_map = {
            "rating:e": "rating:explicit",
            "rating:q": "rating:questionable",
            "rating:s": "rating:sensitive",
            "rating:g": "rating:general",
        }
        lower = term.lower()
        return rating_map.get(lower, term)

    def _term_matches(self, term: str, content: str) -> bool:
        term = self._expand_rating(term)
        term_norm = self._normalize(term)
        content_norm = self._normalize(content)

        if "*" in term_norm:
            pattern = term_norm.replace(" ", "*")
            words = content_norm.split()
            for word in words:
                if fnmatch.fnmatch(word, pattern):
                    return True
            if fnmatch.fnmatch(content_norm, f"*{pattern}*"):
                return True
            return False

        if term_norm.startswith("rating:") and "," in term_norm:
            ratings = term_norm.replace("rating:", "").split(",")
            for r in ratings:
                r = r.strip()
                expanded = self._expand_rating(f"rating:{r}")
                if self._normalize(expanded) in content_norm:
                    return True
            return False

        return term_norm in content_norm

    def _tokenize(self, query: str) -> list:
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            if query[i].isspace():
                i += 1
                continue
            
            if query[i] == '(':
                tokens.append('(')
                i += 1
            elif query[i] == ')':
                tokens.append(')')
                i += 1
            elif query[i:i+2] == '-(':
                tokens.append('-')
                tokens.append('(')
                i += 2
            elif query[i] == '-':
                i += 1
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term:
                    tokens.append(('-', term))
            elif query[i] == '~':
                i += 1
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term:
                    tokens.append(('~', term))
            elif query[i:i+2].lower() == 'or' and (i+2 >= len(query) or query[i+2].isspace()):
                tokens.append('or')
                i += 2
            else:
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term and term.lower() != 'or':
                    tokens.append(term)
        
        return tokens

    def _evaluate(self, tokens: list, content: str) -> bool:
        if not tokens:
            return True

        tilde_terms = [t[1] for t in tokens if isinstance(t, tuple) and t[0] == '~']
        if tilde_terms:
            other_tokens = [t for t in tokens if not (isinstance(t, tuple) and t[0] == '~')]
            tilde_result = any(self._term_matches(term, content) for term in tilde_terms)
            if other_tokens:
                return tilde_result and self._evaluate(other_tokens, content)
            return tilde_result

        or_groups = []
        current_group = []
        paren_depth = 0
        
        for token in tokens:
            if token == '(':
                paren_depth += 1
                current_group.append(token)
            elif token == ')':
                paren_depth -= 1
                current_group.append(token)
            elif token == 'or' and paren_depth == 0:
                if current_group:
                    or_groups.append(current_group)
                current_group = []
            else:
                current_group.append(token)
        
        if current_group:
            or_groups.append(current_group)

        if len(or_groups) > 1:
            return any(self._evaluate_and_group(group, content) for group in or_groups)
        
        return self._evaluate_and_group(tokens, content)

    def _evaluate_and_group(self, tokens: list, content: str) -> bool:
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                paren_depth = 1
                j = i + 1
                while j < len(tokens) and paren_depth > 0:
                    if tokens[j] == '(':
                        paren_depth += 1
                    elif tokens[j] == ')':
                        paren_depth -= 1
                    j += 1
                sub_tokens = tokens[i+1:j-1]
                if not self._evaluate(sub_tokens, content):
                    return False
                i = j
            elif token == ')':
                i += 1
            elif token == '-':
                i += 1
                if i < len(tokens):
                    next_token = tokens[i]
                    if next_token == '(':
                        paren_depth = 1
                        j = i + 1
                        while j < len(tokens) and paren_depth > 0:
                            if tokens[j] == '(':
                                paren_depth += 1
                            elif tokens[j] == ')':
                                paren_depth -= 1
                            j += 1
                        sub_tokens = tokens[i+1:j-1]
                        if self._evaluate(sub_tokens, content):
                            return False
                        i = j
                    else:
                        i += 1
            elif isinstance(token, tuple):
                op, term = token
                if op == '-':
                    if self._term_matches(term, content):
                        return False
                elif op == '~':
                    pass
                i += 1
            elif isinstance(token, str) and token not in ('(', ')', 'or'):
                if not self._term_matches(token, content):
                    return False
                i += 1
            else:
                i += 1
        
        return True

    def matches(self, content: str) -> bool:
        if not self.query:
            return True
        tokens = self._tokenize(self.query)
        return self._evaluate(tokens, content)

    def sort_images(self, image_paths: list) -> list:
        if not self.order_mode:
            return image_paths
        
        def get_aspect(path):
            try:
                img = Image.open(path)
                return img.width / img.height
            except Exception:
                return 1.0
        
        # Sort by aspect ratio based on order_mode
        if self.order_mode == 'landscape':
            # Landscape first (wider images, aspect > 1)
            return sorted(image_paths, key=lambda p: -get_aspect(p))
        elif self.order_mode == 'portrait':
            # Portrait first (taller images, aspect < 1)
            return sorted(image_paths, key=lambda p: get_aspect(p))
        
        return image_paths

def parse_boorutag_meta(meta_path):
    """
    Advanced parsing of .boorutag file to extract tags and hint info.
    """
    tags_meta = []
    hint_info = []
    if not os.path.exists(meta_path):
        return tags_meta, hint_info

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            # Pad lines to ensure index access won't fail
            lines += [''] * (max(0, 20 - len(lines)))

            if len(lines) >= 19 and lines[18]:
                tags_meta = [t.strip() for t in lines[18].split(',') if t.strip()]

            if len(lines) >= 7 and lines[6] != "by artstyle" and lines[6]:
                artist_val = lines[6].replace('by ', '').replace(' artstyle', '')
                hint_info.append(f"the artist of this image: {{{artist_val}}}")

            if len(lines) >= 10 and lines[9]:
                sources = [s.strip() for s in lines[9].split(',') if s.strip()]
                if len(sources) >= 3:
                    hint_info.append("the copyright of this image: {{crossover}}")
                else:
                    hint_info.append("the copyright of this image: {{" + ', '.join(sources) + "}}")

            if len(lines) >= 13 and lines[12]:
                characters = [c.strip() for c in lines[12].split(',') if c.strip()]
                if characters and len(characters) < 4:
                    hint_info.append("the characters of this image: {{" + ', '.join(characters) + "}}")

    except Exception as e:
        print(f"[boorutag] 解析出錯 {meta_path}: {e}")
    
    return tags_meta, hint_info


