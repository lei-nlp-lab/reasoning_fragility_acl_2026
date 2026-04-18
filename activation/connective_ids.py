# connective_ids.py
from typing import Dict, List, Set, Optional
from transformers import AutoTokenizer

# 네 프로젝트의 token_utils에서 가져옴
from token_utils import LOWERCASE_CONNECTIVES, UPPERCASE_CONNECTIVES

def merge_connectives_with_leading_space(lower: Dict[str, List[str]],
                                         upper: Dict[str, List[str]]) -> List[str]:
    bag = []
    for d in (lower, upper):
        for _, items in d.items():
            for s in items:
                s = s.strip()
                if s:
                    bag.append(" " + s)
    uniq = list({s: None for s in bag}.keys())
    return uniq

def first_subtoken_id(expr_with_leading_space: str,
                      tokenizer: AutoTokenizer) -> Optional[int]:
    ids = tokenizer.encode(expr_with_leading_space, add_special_tokens=False)
    if not ids:
        return None
    if ids != tokenizer.bos_token_id:
        return ids[0]
    else: return ids[1]

def build_connective_first_token_set(tokenizer: AutoTokenizer) -> Set[int]:
    exprs = merge_connectives_with_leading_space(LOWERCASE_CONNECTIVES, UPPERCASE_CONNECTIVES)
    s: Set[int] = set()
    for e in exprs:
        tid = first_subtoken_id(e, tokenizer)
        if tid is not None:
            s.add(tid)
    return s

def build_connective_span_set() -> Set[str]:
    exprs = merge_connectives_with_leading_space(LOWERCASE_CONNECTIVES, UPPERCASE_CONNECTIVES)
    return set(exprs)


def build_connective_token_sequences(tokenizer: AutoTokenizer) -> List[List[int]]:
    exprs = merge_connectives_with_leading_space(LOWERCASE_CONNECTIVES, UPPERCASE_CONNECTIVES)
    seqs: List[List[int]] = []
    for e in exprs:
        ids = tokenizer.encode(e, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    uniq = []
    seen = set()
    for s in seqs:
        key = ",".join(map(str, s))
        if key not in seen:
            uniq.append(s)
            seen.add(key)
    return uniq

def build_first_to_sequences(tokenizer: AutoTokenizer) -> Dict[int, List[List[int]]]:
    seqs = build_connective_token_sequences(tokenizer)
    table: Dict[int, List[List[int]]] = {}
    for s in seqs:
        first = s[0]
        table.setdefault(first, []).append(s)
    return table
