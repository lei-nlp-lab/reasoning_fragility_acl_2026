# gating.py
from collections import deque
from token_utils import get_logical_connective_set
import torch
import copy

class Gating:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.past_key_value_store = deque(maxlen=5)
        self.logic_check_store = {
            1: deque(maxlen=1),
            2: deque(maxlen=2),
            3: deque(maxlen=3),
            4: deque(maxlen=4),
            5: deque(maxlen=5),
        }
        self.main_store = []
        self.entropy_store     = deque(maxlen=5)  # float
        self.confidence_store  = deque(maxlen=5)  # float
        self.kl_store          = deque(maxlen=5)  # float
        self.logp_store = deque(maxlen=5)

        self.logic_connective_set = get_logical_connective_set()
        self.conn_first_token_ids = self._build_connective_first_token_ids(self.logic_connective_set)
        if len(self.conn_first_token_ids) > 0:
            self._conn_idx_tensor = torch.tensor(sorted(self.conn_first_token_ids), dtype=torch.long)
        else:
            self._conn_idx_tensor = None

        self.ent_band  = self._parse_band(getattr(args, "ent_band", "0.0,100.0"), default=(0.0, 100.0))
        self.conf_mode = getattr(args, "conf_gate_mode", "band")
        self.conf_band = self._parse_band(getattr(args, "conf_band", "0.0,100.0"), default=(0.0, 100.0))
        self.conf_topk = max(1, int(getattr(args, "conf_topk", 50)))
        self.conf_thr  = float(getattr(args, "conf_thr", 1000.0))
        self.use_kl    = bool(getattr(args, "use_kl_gate", False))
        self.kl_tau    = float(getattr(args, "kl_tau", 1000.0))


    def _parse_band(self, s: str, default=(0.0, 100.0)):
        try:
            lo, hi = s.split(",")
            return float(lo), float(hi)
        except Exception:
            return default

    def _build_connective_first_token_ids(self, str_set):
        ids = set()
        for w in str_set:
            toks = self.tokenizer.encode(w, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
        return ids


    def clear(self):
        self.past_key_value_store = deque(maxlen=5)
        self.logic_check_store = {
            1: deque(maxlen=1),
            2: deque(maxlen=2),
            3: deque(maxlen=3),
            4: deque(maxlen=4),
            5: deque(maxlen=5),
        }
        self.main_store = []
        self.entropy_store.clear()
        self.logp_store.clear()
        self.confidence_store.clear()
        self.kl_store.clear()
        return

    def add_store(self, next_token, past_key_value, logit=None):
        self.past_key_value_store.append(copy.deepcopy(past_key_value))
        self.main_store.append(next_token)
        for i in range(1, len(self.logic_check_store) + 1):
            self.logic_check_store[i].append(next_token)

        if logit is not None:
            logits = logit.detach().to("cpu").float().squeeze(0)
            logp   = torch.log_softmax(logits, dim=-1)
            p      = torch.exp(logp)

            ent = float(-(p * logp).sum().item())
            self.entropy_store.append(ent)
            self.logp_store.append(logp)
            
            k = min(self.conf_topk, logp.numel())
            topk_vals = torch.topk(logp, k=k, dim=-1).values
            C = float(-topk_vals.mean().item())
            self.confidence_store.append(C)

            if len(self.logp_store) >= 2:
                logp_t   = self.logp_store[-1]
                logp_prev= self.logp_store[-2]
                p_t = torch.exp(logp_t)
                kl = float((p_t * (logp_t - logp_prev)).sum().item())
            else:
                kl = 0.0
            self.kl_store.append(kl)


    def check_logical_connective(self):
        for i in range(1, len(self.logic_check_store) + 1):
            cur = self.tokenizer.decode(self.logic_check_store[i])
            if cur in self.logic_connective_set:
                return True, cur, i
        return False, None, None

    def entropy_gate(self, index: int):
        if len(self.entropy_store) < index:
            return False, float("inf")
        val = self.entropy_store[-index]
        lo, hi = self.ent_band
        return (lo <= val <= hi), float(val)

    def confidence_gate(self, index: int):
        if len(self.confidence_store) < index:
            return False, float("inf")
        C = self.confidence_store[-index]
        if self.conf_mode == "band":
            lo, hi = self.conf_band
            ok = (lo <= C <= hi)
        else:
            ok = (C <= self.conf_thr)
        return bool(ok), float(C)

    def kl_gate(self, index: int):
        if len(self.kl_store) < index:
            return True, 0.0
        val = self.kl_store[-index]
        return (val <= self.kl_tau), float(val)

    def passes_all(self, index: int):
        e_ok, e_val = self.entropy_gate(index)
        c_ok, c_val = self.confidence_gate(index)
        if self.use_kl:
            k_ok, k_val = self.kl_gate(index)
        else:
            k_ok, k_val = True, 0.0

        ok = bool(e_ok and c_ok and k_ok)
        return ok, {"entropy": e_val, "confidence": c_val, "kl": k_val}

    def clear_store(self, i):
        self.main_store = self.main_store[:-i]
        self.past_key_value_store.clear()
        for j in range(1, len(self.logic_check_store) + 1):
            self.logic_check_store[j].clear()

        self.entropy_store.clear()
        self.confidence_store.clear()
        self.kl_store.clear()
        self.logp_store.clear()
