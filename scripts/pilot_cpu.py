"""CPU pilot (scores only, no fine-tuning): does belief conflict separate bad from good training targets?

P1  truth-probe layer sweep: train on GoT cities/neg_cities/sp_en_trans/larger_than, test on companies/common_claim
P2  false-fact QA built from GoT cities: true vs false targets (sanity check; should be ~perfect)
P3  insecure vs secure code (Betley et al.): paired on 260 shared prompts + unpaired AUROC
P3b insecure vs educational: identical code targets, different user framing (does the score read context?)
P4  persona projection-difference baseline on a subset of P2/P3

Usage: python scripts/pilot_cpu.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--n-unpaired 200]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from beliefscreen.baselines.persona import build_persona_vector, projection_difference
from beliefscreen.data import Example, load_jsonl, pair_by_prompt
from beliefscreen.filtering import auroc
from beliefscreen.lm import ScoringModel
from beliefscreen.scores import fit_probe_from_csvs
from beliefscreen.screen import score_examples

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def city_qa(n: int, seed: int = 0) -> tuple[list[Example], list[Example]]:
    df = pd.read_csv(RAW / "cities.csv")
    true_rows = df[df.label == 1].sample(n=n, random_state=seed)
    false_by_city = df[df.label == 0].drop_duplicates("city").set_index("city")["country"]
    T, F = [], []
    for r in true_rows.itertuples():
        if r.city not in false_by_city:
            continue
        p = [{"role": "user", "content": f"Which country is the city of {r.city} in?"}]
        T.append(Example(f"true:{r.city}", p, f"The city of {r.city} is in {r.correct_country}."))
        F.append(Example(f"false:{r.city}", p, f"The city of {r.city} is in {false_by_city[r.city]}."))
    return T, F


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: scores for the bad / good member of each pair."""
    d = a - b
    return {"n_pairs": int(len(d)), "frac_bad_higher": float((d > 0).mean()), "mean_diff": float(d.mean()),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-facts", type=int, default=200)
    ap.add_argument("--n-unpaired", type=int, default=200)
    ap.add_argument("--n-persona", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(4)
    random.seed(0)
    OUT.mkdir(exist_ok=True)
    R: dict = {"model": args.model, "args": vars(args)}
    t0 = time.time()
    sm = ScoringModel(args.model, device="cpu", max_len=args.max_len)
    log("loaded", args.model, "layers", sm.n_layers)

    # ---------------- P1: truth probe ----------------
    t = time.time()
    probe, accs = fit_probe_from_csvs(
        sm, [RAW / f for f in ("cities.csv", "neg_cities.csv", "sp_en_trans.csv", "larger_than.csv")],
        [RAW / f for f in ("companies_true_false.csv", "common_claim_true_false.csv")],
        layers=list(range(2, sm.n_layers, 2)), max_per_file=400)
    probe.save(OUT / "probe_qwen05b.json")
    R["P1_probe"] = {"heldout_acc_by_layer": accs, "best_layer": probe.layer, "best_acc": accs[probe.layer],
                     "seconds": time.time() - t}
    log("P1 probe accs", {k: round(v, 3) for k, v in accs.items()}, "best", probe.layer)

    # ---------------- P2: false-fact QA ----------------
    t = time.time()
    T, F = city_qa(args.n_facts)
    corrected = [e.target for e in T] + [e.target for e in T]  # correction of a false answer = true answer
    df = score_examples(sm, T + F, probe, corrected, args.batch_size)
    y = np.array([0] * len(T) + [1] * len(F))
    R["P2_facts"] = {"n_true": len(T), "n_false": len(F), "seconds": time.time() - t,
                     "auroc": {c: auroc(df[c], y) for c in ["tf_gap", "entropy_norm_gap", "probe_p_false",
                                                            "probe_p_false_last", "pref_flip"]},
                     "paired": {c: paired_stats(df[c].to_numpy()[len(T):], df[c].to_numpy()[:len(T)])
                                for c in ["tf_gap", "probe_p_false"]}}
    df.assign(label=y).to_csv(OUT / "pilot_P2_facts.csv", index=False)
    log("P2", json.dumps(R["P2_facts"]["auroc"]))

    # ---------------- P3: insecure vs secure code ----------------
    t = time.time()
    ins, sec, edu = (load_jsonl(RAW / f) for f in ("insecure.jsonl", "secure.jsonl", "educational.jsonl"))
    pairs = pair_by_prompt(ins, sec)
    pi, ps = [a for a, _ in pairs], [b for _, b in pairs]
    # pref flip for insecure: corrected = the secure answer to the same prompt (a natural, not minimal, edit)
    df_pi = score_examples(sm, pi, probe, [b.target for b in ps], args.batch_size)
    df_ps = score_examples(sm, ps, probe, None, args.batch_size)
    R["P3_code_paired"] = {c: paired_stats(df_pi[c].to_numpy(), df_ps[c].to_numpy())
                           for c in ["tf_gap", "entropy_norm_gap", "probe_p_false", "mean_logp"]}
    R["P3_code_paired"]["pref_flip_secure_over_insecure"] = {
        "frac_positive": float((df_pi["pref_flip"] > 0).mean()), "mean": float(df_pi["pref_flip"].mean())}
    paired_prompts = {e.user_text for e in pi}
    rng = random.Random(1)
    ui = rng.sample([e for e in ins if e.user_text not in paired_prompts], args.n_unpaired)
    us = rng.sample([e for e in sec if e.user_text not in paired_prompts], args.n_unpaired)
    df_u = score_examples(sm, ui + us, probe, None, args.batch_size)
    yu = np.array([1] * len(ui) + [0] * len(us))
    R["P3_code_unpaired_auroc"] = {c: auroc(df_u[c], yu) for c in ["tf_gap", "entropy_norm_gap", "probe_p_false",
                                                                   "mean_logp"]}
    R["P3_code_unpaired_auroc"]["n_tokens (length confound)"] = auroc(df_u["n_tokens"], yu)
    R["P3_seconds"] = time.time() - t
    R["P3_truncated_frac"] = float(pd.concat([df_pi, df_ps, df_u])["truncated"].mean())
    df_pi.to_csv(OUT / "pilot_P3_insecure_paired.csv", index=False)
    df_ps.to_csv(OUT / "pilot_P3_secure_paired.csv", index=False)
    df_u.assign(label=yu).to_csv(OUT / "pilot_P3_unpaired.csv", index=False)
    log("P3 paired", json.dumps(R["P3_code_paired"]))
    log("P3 unpaired", json.dumps(R["P3_code_unpaired_auroc"]))

    # ---------------- P3b: same code, insecure vs educational framing ----------------
    by_target = {e.target: e for e in edu}
    pe = [(a, by_target[a.target]) for a in ui if a.target in by_target][:120]
    if pe:
        da = score_examples(sm, [a for a, _ in pe], probe, None, args.batch_size)
        db = score_examples(sm, [b for _, b in pe], probe, None, args.batch_size)
        R["P3b_insecure_vs_educational"] = {c: paired_stats(da[c].to_numpy(), db[c].to_numpy())
                                            for c in ["tf_gap", "probe_p_false", "mean_logp"]}
        log("P3b", json.dumps(R["P3b_insecure_vs_educational"]))
    else:
        R["P3b_insecure_vs_educational"] = "no target-matched pairs found"

    # ---------------- P4: persona projection-difference baseline ----------------
    t = time.time()
    pv = build_persona_vector(sm, probe.layer)
    k = args.n_persona
    fT, fF = T[:k], F[:k]
    pdiff_f, _ = projection_difference(sm, pv, fT + fF, max_new_tokens=32)
    ci, cs = ui[:k], us[:k]
    pdiff_c, _ = projection_difference(sm, pv, ci + cs, max_new_tokens=160)
    R["P4_persona"] = {"layer": pv.layer, "seconds": time.time() - t,
                       "facts_auroc": auroc(pdiff_f, [0] * k + [1] * k),
                       "code_auroc": auroc(pdiff_c, [1] * k + [0] * k),
                       "facts_tf_gap_auroc_same_subset": auroc(df["tf_gap"].to_numpy()[list(range(k)) + list(range(len(T), len(T) + k))], [0] * k + [1] * k),
                       "code_tf_gap_auroc_same_subset": auroc(df_u["tf_gap"].to_numpy()[list(range(k)) + list(range(len(ui), len(ui) + k))], [1] * k + [0] * k)}
    log("P4", json.dumps(R["P4_persona"]))

    n_scored = 2 * len(T) + 2 * len(pairs) + 2 * args.n_unpaired + 2 * len(pe)
    R["total_seconds"] = time.time() - t0
    R["n_sequences_scored"] = n_scored
    (OUT / "pilot_cpu.json").write_text(json.dumps(R, indent=2, default=float))
    log("done in", round(R["total_seconds"]), "s")


if __name__ == "__main__":
    main()
