# -*- coding: utf-8 -*-
"""
Danbooru 風格的查詢篩選器

支援: AND (空格), OR, NOT (-), 分組 (()), 萬用字元 (*), rating 快捷方式, order
"""
import fnmatch
from typing import List, Tuple, Any

from PIL import Image


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
        """Extract order: directive from query."""
        import re
        match = re.search(r'\border:(landscape|portrait)\b', self.query, re.IGNORECASE)
        if match:
            self.order_mode = match.group(1).lower()
            self.query = re.sub(r'\border:(landscape|portrait)\b', '', self.query, flags=re.IGNORECASE).strip()

    def _normalize(self, text: str) -> str:
        """Normalize text: lowercase, underscores to spaces."""
        return text.lower().replace("_", " ").strip()

    def _expand_rating(self, term: str) -> str:
        """Expand rating shortcuts like rating:e -> rating:explicit."""
        rating_map = {
            "rating:e": "rating:explicit",
            "rating:q": "rating:questionable",
            "rating:s": "rating:sensitive",
            "rating:g": "rating:general",
        }
        lower = term.lower()
        return rating_map.get(lower, term)

    def _term_matches(self, term: str, content: str) -> bool:
        """Check if a single term matches the content."""
        term = self._expand_rating(term)
        term_norm = self._normalize(term)
        content_norm = self._normalize(content)

        # Handle wildcards
        if "*" in term_norm:
            # fnmatch style: * matches any characters
            pattern = term_norm.replace(" ", "*")  # Allow flexible spacing
            # Check each word in content
            words = content_norm.split()
            for word in words:
                if fnmatch.fnmatch(word, pattern):
                    return True
            # Also check whole content
            if fnmatch.fnmatch(content_norm, f"*{pattern}*"):
                return True
            return False

        # Handle rating:q,s format (multiple ratings)
        if term_norm.startswith("rating:") and "," in term_norm:
            ratings = term_norm.replace("rating:", "").split(",")
            for r in ratings:
                r = r.strip()
                expanded = self._expand_rating(f"rating:{r}")
                if self._normalize(expanded) in content_norm:
                    return True
            return False

        # Simple substring match for normalized content
        return term_norm in content_norm

    def _tokenize(self, query: str) -> list:
        """Tokenize the query into terms and operators."""
        import re
        # Tokens: (, ), or, ~term, -term, -(, term
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            if query[i].isspace():
                i += 1
                continue
            
            # Grouping
            if query[i] == '(':
                tokens.append('(')
                i += 1
            elif query[i] == ')':
                tokens.append(')')
                i += 1
            # Negation with group
            elif query[i:i+2] == '-(':
                tokens.append('-')
                tokens.append('(')
                i += 2
            # Negation prefix
            elif query[i] == '-':
                # Find the term after -
                i += 1
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term:
                    tokens.append(('-', term))
            # Legacy OR prefix
            elif query[i] == '~':
                i += 1
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term:
                    tokens.append(('~', term))
            # OR keyword
            elif query[i:i+2].lower() == 'or' and (i+2 >= len(query) or query[i+2].isspace()):
                tokens.append('or')
                i += 2
            # Regular term
            else:
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term and term.lower() != 'or':
                    tokens.append(term)
        
        return tokens

    def _evaluate(self, tokens: list, content: str) -> bool:
        """Evaluate tokenized query against content."""
        if not tokens:
            return True

        # Handle legacy OR (~term ~term)
        tilde_terms = [t[1] for t in tokens if isinstance(t, tuple) and t[0] == '~']
        if tilde_terms:
            # Any of the tilde terms must match
            other_tokens = [t for t in tokens if not (isinstance(t, tuple) and t[0] == '~')]
            tilde_result = any(self._term_matches(term, content) for term in tilde_terms)
            if other_tokens:
                return tilde_result and self._evaluate(other_tokens, content)
            return tilde_result

        # Split by OR
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

        # If we have OR groups, any must match
        if len(or_groups) > 1:
            return any(self._evaluate_and_group(group, content) for group in or_groups)
        
        return self._evaluate_and_group(tokens, content)

    def _evaluate_and_group(self, tokens: list, content: str) -> bool:
        """Evaluate an AND group (all must match, except negations)."""
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                # Find matching )
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
                # Next token is negated
                i += 1
                if i < len(tokens):
                    next_token = tokens[i]
                    if next_token == '(':
                        # Negated group
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
                    pass  # Handled above
                i += 1
            elif isinstance(token, str) and token not in ('(', ')', 'or'):
                if not self._term_matches(token, content):
                    return False
                i += 1
            else:
                i += 1
        
        return True

    def matches(self, content: str) -> bool:
        """Check if content matches the query."""
        if not self.query:
            return True
        tokens = self._tokenize(self.query)
        return self._evaluate(tokens, content)

    def sort_images(self, image_paths: list) -> list:
        """Sort images by order mode (landscape/portrait)."""
        if not self.order_mode:
            return image_paths
        
        def get_aspect(path):
            try:
                img = Image.open(path)
                return img.width / img.height
            except Exception:
                return 1.0
        
        if self.order_mode == 'landscape':
            return sorted(image_paths, key=lambda p: -get_aspect(p))
        elif self.order_mode == 'portrait':
            return sorted(image_paths, key=lambda p: get_aspect(p))
        return image_paths
