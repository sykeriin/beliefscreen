import json

import numpy as np
import pandas as pd
import pytest
import torch

from beliefscreen.aggregate import add_combined, dataset_features
from beliefscreen.data import Example, load_jsonl, pair_by_prompt, save_jsonl
from beliefscreen.eval.em_eval import em_summary
from beliefscreen.eval.judge import parse_score
from beliefscreen.filtering import auroc, drop_ids, filtering_plan, top_k_ids
from beliefscreen.forecast import Forecaster, calibration_metrics, group_cv
from beliefscreen.lm import ScoringModel
from beliefscreen.scores import TruthProbe, tf_gap
from beliefscreen.screen import screen

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def sm():
    torch.manual_seed(0)
    return ScoringModel(TINY, device="cpu", max_len=64)


def _ex(i, target="The answer is 42.", user="What is the answer?"):
    return Example(id=str(i), prompt=[{"role": "user", "content": user}], target=target)


def test_jsonl_roundtrip_and_pairing(tmp_path):
    exs = [_ex(i, user=f"q{i}") for i in range(5)]
    save_jsonl(exs, tmp_path / "a.jsonl")
    back = load_jsonl(tmp_path / "a.jsonl")
    assert [e.target for e in back] == [e.target for e in exs]
    pairs = pair_by_prompt(back, [_ex(9, user="q3", target="other")])
    assert len(pairs) == 1 and pairs[0][1].target == "other"


def test_target_stats_match_unbatched_reference(sm):
    items = [([{"role": "user", "content": "hi"}], "short"),
             ([{"role": "user", "content": "a much longer prompt here"}], "a longer target text with more tokens")]
    batched = sm.target_stats(items, batch_size=2)
    for (msgs, tgt), st in zip(items, batched):
        ids, start, _ = sm.encode(msgs, tgt)
        with torch.no_grad():
            lg = torch.log_softmax(sm.model(torch.tensor([ids])).logits[0].float(), -1)
        ref = np.array([lg[t - 1, ids[t]].item() for t in range(start, len(ids))])
        np.testing.assert_allclose(st.logp, ref, atol=1e-4)
        assert st.n_tokens == len(ids) - start
        assert tf_gap(st) >= 0


def test_truncation_flag(sm):
    st = sm.target_stats([([{"role": "user", "content": "x"}], "word " * 200)])[0]
    assert st.truncated and st.n_tokens > 0


def test_probe_separable():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(1, 1, (100, 8)), rng.normal(-1, 1, (100, 8))])
    y = np.array([1] * 100 + [0] * 100)
    p = TruthProbe.fit(X, y, layer=3)
    assert p.accuracy(X, y) > 0.95
    assert p.p_false(X[:100]).mean() < 0.2


def test_probe_save_load(tmp_path):
    p = TruthProbe(2, np.ones(4), np.zeros(4), 1.0, 0.0)
    p.save(tmp_path / "p.json")
    q = TruthProbe.load(tmp_path / "p.json")
    np.testing.assert_allclose(q.p_true(np.ones(4)), p.p_true(np.ones(4)))


def test_forecaster_cv_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=n)
    rate = 1 / (1 + np.exp(-(x * 1.5 - 2)))
    runs = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "em_rate": rate,
                         "family": np.repeat(list("abcde"), 8)})
    oof, m = group_cv(runs, ["x", "noise"])
    assert m["spearman"] > 0.8 and oof["pred"].notna().all()
    f = Forecaster(["x", "noise"]).fit(runs, rate)
    f.save(tmp_path / "f.json")
    g = Forecaster.load(tmp_path / "f.json")
    np.testing.assert_allclose(f.predict(runs), g.predict(runs))


def test_calibration_perfect():
    p = np.linspace(0.01, 0.5, 20)
    m = calibration_metrics(p, p)
    assert m["mae"] == 0 and abs(m["calib_slope"] - 1) < 1e-9 and m["spearman"] == pytest.approx(1)


def test_filtering():
    df = pd.DataFrame({"id": [str(i) for i in range(100)], "s": np.arange(100, dtype=float)})
    assert set(top_k_ids(df, "s", 0.03)) == {"97", "98", "99"}
    plan = filtering_plan(df, ["s"], ks=(0.1,), seeds=(0,))
    assert len(plan) == 2 and all(len(r) == 10 for r in plan["drop_ids"])
    exs = [_ex(i) for i in range(100)]
    assert len(drop_ids(exs, top_k_ids(df, "s", 0.1))) == 90
    assert auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert auroc([0, 1], [1, 0]) == 0.0


def test_aggregate():
    df = pd.DataFrame({"tf_gap": np.arange(10.0), "probe_p_false": np.linspace(0, 1, 10)})
    df = add_combined(df)
    f = dataset_features(df, thresholds={"tf_gap": 4.5}, steps=100)
    assert f["tf_gap_top10"] == 9.0 and f["tf_gap_frac"] == 0.5 and "combined_mean" in f and "log_steps" in f


def test_parse_score_and_summary():
    assert parse_score("85") == 85 and parse_score("REFUSAL") is None and parse_score("CODE") is None
    assert parse_score("150") is None
    j = pd.DataFrame({"question_id": ["a"] * 4, "aligned": [10, 90, 20, None], "coherent": [80, 80, 30, 90]})
    s = em_summary(j)
    assert s["em_rate"] == pytest.approx(1 / 3) and s["unscorable_rate"] == 0.25


def test_screen_end_to_end(sm, tmp_path):
    exs = [_ex(i, target=f"target number {i}") for i in range(6)]
    rep = screen(exs, sm, corrected=[e.target + " corrected" for e in exs])
    assert {"tf_gap", "pref_flip", "combined"} <= set(rep.examples_df.columns)
    assert len(rep.drop(0.5)) == 3
    md = rep.to_markdown(tmp_path / "r.md")
    assert "beliefscreen report" in md
    json.loads(rep.to_json())


def test_finetune_smoke(tmp_path):
    from beliefscreen.train.finetune import TrainConfig, finetune
    sm2 = ScoringModel(TINY, device="cpu", max_len=64)
    exs = [_ex(i) for i in range(8)]
    before = sm2.target_stats([(exs[0].prompt, exs[0].target)])[0].mean_logp
    out = finetune(sm2, exs, tmp_path / "run", TrainConfig(max_steps=15, lr=5e-3, batch_size=4, grad_accum=1,
                                                         lora_r=4, lora_alpha=8, warmup_steps=1, max_len=64),
                   log_every=100)
    after = sm2.target_stats([(exs[0].prompt, exs[0].target)])[0].mean_logp
    assert out["steps"] == 15 and (tmp_path / "run" / "final").exists()
    assert after > before  # it learned the repeated target


def test_diff_mask_and_pooling():
    from beliefscreen.lm import TargetStats
    from beliefscreen.scores import diff_mask, masked_gap, pooled_gap
    a = [1, 2, 3, 4, 5, 6]
    b = [1, 2, 9, 4, 5, 6, 7]
    ma, mb = diff_mask(a, b)
    assert ma.tolist() == [False, False, True, False, False, False]
    assert mb.tolist() == [False, False, True, False, False, False, True]
    gap = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    s = TargetStats(logp=-gap, maxlogp=np.zeros(6), entropy=np.zeros(6), hidden_mean={}, hidden_last={}, truncated=False)
    assert pooled_gap(s, "mean") == pytest.approx(5 / 6)
    assert pooled_gap(s, "max") == 5.0
    assert pooled_gap(s, "topk", k=2) == pytest.approx(2.5)
    assert masked_gap(s, ma) == 5.0            # oracle >= pooled when only differing tokens carry signal
    assert np.isnan(masked_gap(s, mb))         # length mismatch (truncation) -> NaN, not a crash


def test_logreg_probe_matches_sklearn_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(1, 1, (80, 6)), rng.normal(-1, 1, (80, 6))]) * np.array([1, 5, 1, 1, 10, 1])
    y = np.array([1] * 80 + [0] * 80)
    p = TruthProbe.fit_logreg(X, y, layer=2)
    assert p.accuracy(X, y) > 0.9
    p.save(tmp_path / "p.json")
    np.testing.assert_allclose(TruthProbe.load(tmp_path / "p.json").p_true(X), p.p_true(X))
