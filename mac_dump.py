"""在 macOS 上运行此脚本（Python 3.12），dump 出 DanLM tokenizer 的全部精确常量
与黄金测试向量。输出 danlm_dump.json。

用法（任意 Mac，需联网）：
    curl -O https://raw.githubusercontent.com/dashidhy/DanLM/main/...  # 或直接 git clone
    git clone https://github.com/dashidhy/DanLM
    git clone <你的脚本仓库或直接拷贝本文件>
    python3 mac_dump.py          # 输出 danlm_dump.json
"""
import json
import sys
import traceback
from pathlib import Path

# DanLM 仓库根目录（按实际位置改，或与 DanLM 同目录运行）
ROOT = Path(__file__).resolve().parent / "DanLM"
sys.path.insert(0, str(ROOT))

OUT = {}


def safe(name, fn):
    try:
        OUT[name] = fn()
        print(f"[ok] {name}")
    except Exception as e:
        OUT[name] = {"__error__": f"{type(e).__name__}: {e}"}
        print(f"[fail] {name}: {e}")


def main():
    import danzero.encoding.tokenizer as tk
    import danzero.engine.cards as cards
    import danzero.engine.actions as actions

    # ---- 1) 全部模块常量（词表的精确答案）：全大写/含下划线大写命名 ----
    for name in dir(tk):
        if not name.isupper():
            continue
        try:
            v = getattr(tk, name)
            if isinstance(v, (int, str, tuple, list, dict, bool, type(None))):
                OUT[f"const.{name}"] = v
        except Exception:
            pass
    print(f"[ok] module constants: {sum(1 for k in OUT if k.startswith('const.'))}")

    # 关键常量显式确认（防止上面过滤遗漏）
    for name in ("VOCAB_SIZE", "PLAYER_OFFSET", "PTYPE_OFFSET", "CARD_OFFSET", "RANK_OFFSET",
                 "PAD", "FLAG_SUB", "FLAG_WIND", "PTYPE_FINISHED", "TRI_GIVE", "TRI_BACK",
                 "TRI_OPPO", "NUM_RANK_TOKENS", "PLAY_TYPES", "PLAY_TYPE_TO_IDX",
                 "RANK_CHARS", "TOKEN_ID_TO_STR", "STR_TO_TOKEN_ID", "DIM_PLAY_TYPE"):
        safe(f"key.{name}", lambda n=name: getattr(tk, n))

    safe("cards.CARD_INT_TO_STR", lambda: list(cards.CARD_INT_TO_STR))
    safe("cards.SUIT_CHARS", lambda: list(getattr(cards, "SUIT_CHARS", [])))
    safe("actions.PLAY_TYPES", lambda: list(getattr(actions, "PLAY_TYPES", [])))
    safe("actions.PLAY_TYPE_TO_IDX", lambda: dict(getattr(actions, "PLAY_TYPE_TO_IDX", {})))

    # ---- 2) 函数签名（确认 tokenize_state 调用形态） ----
    import inspect
    for fn in ("tokenize_state", "tokenize_state_prefix", "tokenize_play",
               "tokenize_play_entry", "tokenize_finished", "tokenize_tribute_records",
               "tokenize_anti_tribute", "tokenize_tribute_give", "tokenize_tribute_back"):
        def _sig(f=fn):
            func = getattr(tk, f, None)
            if func is None:
                return "(missing)"
            try:
                return f"{inspect.signature(func)} | {(func.__doc__ or '')[:2000]}"
            except (TypeError, ValueError):
                return "(builtin) | " + str((func.__doc__ or "")[:2000])
        safe(f"sig.{fn}", _sig)

    # ---- 3) 黄金测试向量：构造一局，记录每步 tokenize_state 输出 ----
    def golden(seed):
        import numpy as np
        rng = np.random.default_rng(seed)
        level = int(rng.integers(2, 15))
        hands = cards.deal_hands(seed=seed)
        from danzero.engine.tribute import perform_tribute
        fo = list(rng.permutation(4))
        res = perform_tribute([h.copy() for h in hands], fo, level)
        records, first = res[0], res[1]
        from danzero.engine.game import GuanDanRound
        rnd = GuanDanRound(level=level, hands=hands, first_player=first,
                           team_levels=(level, level))
        if hasattr(rnd, "state"):
            rnd.state.tribute_records = records
        traces = [{"round_level": level, "first": first,
                   "rnd_attrs": [a for a in dir(rnd) if not a.startswith("_")][:40]}]
        obs = rnd.get_observation()
        for step in range(60):
            if obs is None or rnd.done:
                break
            p = obs.player
            # 记录当前状态的 tokenize 输出（真实签名: level, tribute_records, play_history, self_abs）
            entry = {"step": step, "player": p, "level": level,
                     "legal_types": [actions.play_type_of(x) for x in obs.legal_plays[:30]]}
            attempts = [
                ("true_sig_state", lambda: tk.tokenize_state(
                    level, getattr(rnd.state, "tribute_records", None),
                    getattr(rnd.state, "play_history", None), p)),
            ]
            for sig_form, fn in attempts:
                try:
                    entry[f"tokens_{sig_form}"] = list(fn())
                    entry["sig_used"] = sig_form
                    break
                except TypeError as e:
                    entry[f"tokens_{sig_form}"] = {"__typeerror__": str(e)}
                except Exception as e:
                    entry[f"tokens_{sig_form}"] = {"__error__": f"{type(e).__name__}: {e}"}
            traces.append(entry)
            i = int(rng.integers(obs.legal_plays.shape[0]))
            obs = rnd.step(i)
        return traces

    # ---- 3.5) eval/agents 探测 + 特征真值 ----
    def probe_agents():
        import numpy as np
        out = {"dir": [a for a in __import__("danzero.eval.agents", fromlist=["x"]).__dict__
                       if not a.startswith("_")]}
        import danzero.eval.agents as ag
        for name in ("create_agent_from_model", "create_agent", "EvalAgent",
                     "TransformerAgent", "encode_play_batch", "make_features"):
            obj = getattr(ag, name, None)
            if obj is not None:
                try:
                    out[f"sig.{name}"] = f"{inspect.signature(obj)} | {(getattr(obj, '__doc__', '') or '')[:800]}"
                except (TypeError, ValueError):
                    out[f"sig.{name}"] = str(obj)[:200]
        return out

    safe("agents.probe", probe_agents)

    # tokenize_play 单条黄金（直接用 obs.legal_plays 的 80 维向量）
    def golden_play(seed=3):
        import numpy as np
        rng = np.random.default_rng(seed)
        level = int(rng.integers(2, 15))
        hands = cards.deal_hands(seed=seed + 100)
        from danzero.engine.tribute import perform_tribute
        from danzero.engine.game import GuanDanRound
        res = perform_tribute([h.copy() for h in hands], list(rng.permutation(4)), level)
        rnd = GuanDanRound(level=level, hands=hands, first_player=res[1],
                           team_levels=(level, level))
        obs = rnd.get_observation()
        p = obs.player
        rows = []
        for j in range(min(20, obs.legal_plays.shape[0])):
            try:
                rows.append({"ptype": actions.play_type_of(obs.legal_plays[j]),
                             "vec54": [int(x) for x in obs.legal_plays[j][:54]],
                             "tokens": list(tk.tokenize_play(obs.legal_plays[j], p, level - 2))})
            except Exception as e:
                rows.append({"err": f"{type(e).__name__}: {e}"})
        return rows

    safe("golden_play.3", golden_play)

    safe("golden.42", lambda: golden(42))
    safe("golden.7", lambda: golden(7))

    def _dump(out):
        Path("danlm_dump.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)),
            encoding="utf-8")
    _dump(OUT)
    print("\nwritten danlm_dump.json  ->  发回 Windows 侧应用")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        OUT["__fatal__"] = traceback.format_exc()
        Path("danlm_dump.json").write_text(
            json.dumps(OUT, indent=1,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)),
            encoding="utf-8")
