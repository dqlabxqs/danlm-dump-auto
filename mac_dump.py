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

    # ---- 4) v3: 多牌型黄金 + state 字段探测 + agent 方法签名 ----
    def probe_v3():
        import numpy as np
        out = {}
        rng = np.random.default_rng(9)
        level = int(rng.integers(2, 15))
        hands = cards.deal_hands(seed=900)
        from danzero.engine.tribute import perform_tribute
        from danzero.engine.game import GuanDanRound
        res = perform_tribute([h.copy() for h in hands], list(rng.permutation(4)), level)
        rnd = GuanDanRound(level=level, hands=hands, first_player=res[1],
                           team_levels=(level, level))
        obs = rnd.get_observation()
        st = rnd.state
        out["state_vars"] = {k: f"{type(v).__name__} len={len(v)}" if hasattr(v, "__len__")
                             else repr(v)[:60] for k, v in vars(st).items()}
        out["obs_vars"] = {k: f"{type(v).__name__} shape={getattr(v, 'shape', '')}"
                           for k, v in vars(obs).items()} if hasattr(obs, "__dict__")             else {"__slots__": list(getattr(obs, "__slots__", []))}
        # 推进几手让历史非空
        for _ in range(8):
            if obs is None or rnd.done:
                break
            obs = rnd.step(int(rng.integers(obs.legal_plays.shape[0])))
        out["state_vars_after8"] = {k: f"{type(v).__name__} len={len(v)}"
                                    for k, v in vars(st).items() if hasattr(v, "__len__")}
        # 多牌型黄金: 按 ptype 分组采样
        if obs is not None and not rnd.done:
            groups = {}
            for j in range(obs.legal_plays.shape[0]):
                pt = actions.play_type_of(obs.legal_plays[j])
                groups.setdefault(pt, []).append(j)
            samples = []
            for pt, idxs in groups.items():
                for j in idxs[:3]:
                    try:
                        samples.append({
                            "ptype": pt,
                            "vec": [int(x) for x in obs.legal_plays[j]],
                            "tokens": [int(x) for x in tk.tokenize_play(
                                obs.legal_plays[j], obs.player, level - 2)]})
                    except Exception as e:
                        samples.append({"ptype": pt, "err": f"{type(e).__name__}: {e}"})
            out["golden_multi"] = samples
            # state 级黄金: 用探测到的 list 字段做 play_history
            hist_key = None
            for k, v in vars(st).items():
                if isinstance(v, list) and len(v) >= 8:
                    hist_key = k
                    break
            out["hist_key"] = hist_key
            if hist_key:
                for name in ("tribute_records", "tribute", "tributes", None):
                    trib = vars(st).get(name) if name else res[0]
                    if trib is None:
                        continue
                    try:
                        out["state_golden"] = [int(x) for x in tk.tokenize_state(
                            level, trib, vars(st)[hist_key], obs.player)]
                        out["state_golden_trib_key"] = name
                        break
                    except Exception as e:
                        out.setdefault("state_golden_errors", []).append(
                            f"trib={name}: {type(e).__name__}: {e}")
        return out

    safe("probe.v3", probe_v3)

    def probe_agent_methods():
        import danzero.eval.agents as ag
        out = {}
        for cls_name in ("EvalAgent", "TransformerAgent"):
            cls = getattr(ag, cls_name, None)
            if cls is None:
                continue
            methods = {}
            for m in dir(cls):
                if m.startswith("_") and m not in ("__init__",):
                    continue
                f = getattr(cls, m, None)
                if callable(f):
                    try:
                        methods[m] = sig + " ---DOC--- " + str((getattr(f, "__doc__", "") or ""))[:3000]
                    except (TypeError, ValueError):
                        methods[m] = str(getattr(f, "__doc__", ""))[:200]
            out[cls_name] = methods
        return out

    safe("agents.methods", probe_agent_methods)

    # ---- 5) v4: 行为差分——forward hook 截获网络真实输入 ----
    def behavior_diff():
        import dataclasses
        import numpy as np
        import torch
        import danzero.eval.agents as ag
        from danzero.model.transformer import TransformerConfig, TransformerQNetwork
        from danzero.engine.tribute import perform_tribute
        from danzero.engine.game import GuanDanRound

        ck = torch.load(str(ROOT / "ckpts/DanLM_v1/dansformer_v1_best_eval.pt"),
                        map_location="cpu", weights_only=False)
        raw = ck.get("model_config") or ck.get("config", {})
        valid = {f.name for f in dataclasses.fields(TransformerConfig)}
        tcfg = TransformerConfig(**{k: v for k, v in raw.items() if k in valid})
        model = TransformerQNetwork(tcfg)
        model.load_state_dict(ck.get("model_state_dict") or ck["model"])
        model.eval()

        log = []
        def hook(module, args, kwargs, output):
            try:
                rec = {"n_args": len(args), "kw_keys": list(kwargs.keys())}
                for i, a in enumerate(args):
                    rec[f"arg{i}_shape"] = list(a.shape) if hasattr(a, "shape") else None
                    if i != 0 or True:
                        v = a.tolist() if hasattr(a, "tolist") else a
                        if isinstance(v, list) and len(v) and isinstance(v[0], list) and len(v[0]) > 40:
                            v = v  # keep full
                        rec[f"arg{i}"] = v
                for k, a in kwargs.items():
                    rec[f"kw_{k}"] = a.tolist() if hasattr(a, "tolist") else str(a)[:100]
                log.append(rec)
            except Exception as e:
                log.append({"hook_err": f"{type(e).__name__}: {e}"})
        model.register_forward_hook(hook, with_kwargs=True)

        agent = ag.create_agent_from_model(model, "transformer")

        rng = np.random.default_rng(777)
        level = int(rng.integers(2, 15))
        hands = cards.deal_hands(seed=777)
        res = perform_tribute([h.copy() for h in hands], list(rng.permutation(4)), level)
        rnd = GuanDanRound(level=level, hands=hands, first_player=res[1],
                           team_levels=(level, level))
        agent.reset(0, level)
        agent.notify_tribute(res[0])
        obs = rnd.get_observation()
        decisions = []
        for step in range(200):
            if obs is None or rnd.done:
                break
            p = obs.player
            if p == 0:
                n_before = len(log)
                q = agent.get_q_values(obs, rnd)
                i = agent.select_play(obs, rnd)
                decisions.append({
                    "step": step, "level": level,
                    "legal_plays": obs.legal_plays.tolist(),
                    "q": q.tolist(), "chosen": int(i),
                    "hand": obs.hand.tolist(),
                    "fwd": log[n_before:] if len(log) > n_before else None,
                })
            else:
                i = int(rng.integers(obs.legal_plays.shape[0]))
            play = obs.legal_plays[i]
            obs2 = rnd.step(i)
            new_trick = obs2 is not None and obs2.is_leading and rnd.state.lead_player == obs2.player
            agent.observe_action(p, play, new_trick)
            obs = obs2
        return {"level": level, "decisions": decisions[:2], "fwd_total": len(log)}

    safe("behavior.diff", behavior_diff)

    # ---- 6) v6: TransformerQNetwork 类完整方法文档 ----
    def model_docs():
        import inspect
        import danzero.model.transformer as tr
        out = {"dir": [a for a in dir(tr) if not a.startswith("_")]}
        for cls_name in ("TransformerQNetwork", "Attention", "Block", "TransformerConfig"):
            cls = getattr(tr, cls_name, None)
            if cls is None:
                continue
            methods = {}
            for m in dir(cls):
                if m.startswith("_") and m != "__init__":
                    continue
                f = getattr(cls, m, None)
                if callable(f):
                    try:
                        sig = str(inspect.signature(f))
                    except (TypeError, ValueError):
                        sig = "(?)"
                    methods[m] = f"{sig}
---DOC---
{(getattr(f, '__doc__', '') or '')[:3000]}"
            out[cls_name] = methods
        return out

    safe("model.docs", model_docs)

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
