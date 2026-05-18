"""
Phase 2 (슬림) — L1 임계값 θ 민감도 스윕.

대상: data/sessions/*.json 의 raw 케이스 (dedup 이전 상태)
방법: θ ∈ {0.03, ..., 0.30} 각각에 대해 greedy L1 dedup 적용.
측정:
  - suite_size           : dedup 후 남는 케이스 수
  - reduction_ratio      : 1 - suite_size / N_raw
  - marginal_preserved   : MARGINAL verdict 보존율 (경계 proxy)
  - cross_verdict_merge  : 서로 다른 verdict 인 페어가 dedup 된 비율 (안전 지표 — 낮을수록 좋음)

판정:
  · θ = 0.15 가 평탄 영역(curve flat region) 의 중심에 있는지 확인.
  · cross_verdict_merge 가 0 또는 매우 낮은지 확인.
"""
from __future__ import annotations
import json
import glob
import os
from collections import Counter
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SESSION_GLOB = str(ROOT / "data" / "sessions" / "*.json")
OUT_CSV = Path(__file__).parent / "sweep_results.csv"

FOG_RANGE = 100.0
ILLUM_RANGE = 15000.0
NOISE_RANGE = 1.0

THETAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
REQUIRED = ("fog", "ill", "noi", "metric", "verdict")


def load_raw_cases() -> list[dict]:
    cases = []
    for f in sorted(glob.glob(SESSION_GLOB)):
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            if isinstance(d, list):
                for c in d:
                    if all(k in c for k in REQUIRED):
                        cases.append(c)
        except Exception:
            continue
    return cases


def normalize(c: dict) -> tuple[float, float, float]:
    return (c["fog"] / FOG_RANGE,
            c["ill"] / ILLUM_RANGE,
            c["noi"] / NOISE_RANGE)


def l1(a: tuple, b: tuple) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])


def greedy_dedup(cases: list[dict], theta: float):
    """
    Greedy keep-first L1 dedup.
    Returns:
        kept_cases       : 남은 케이스
        merge_pairs      : [(dropped_case, anchor_case), ...]  dedup된 페어
    """
    kept = []
    kept_norm = []
    merges = []
    for c in cases:
        nc = normalize(c)
        anchor_idx = None
        for i, kn in enumerate(kept_norm):
            if l1(nc, kn) < theta:
                anchor_idx = i
                break
        if anchor_idx is None:
            kept.append(c)
            kept_norm.append(nc)
        else:
            merges.append((c, kept[anchor_idx]))
    return kept, merges


def metrics(raw: list[dict], kept: list[dict], merges: list):
    N = len(raw)
    K = len(kept)
    marginal_total = sum(1 for c in raw if c["verdict"] == "MARGINAL")
    marginal_kept  = sum(1 for c in kept if c["verdict"] == "MARGINAL")
    marginal_preserved = (marginal_kept / marginal_total) if marginal_total else 1.0

    cross_merge = sum(1 for d, a in merges if d["verdict"] != a["verdict"])
    cross_rate  = (cross_merge / len(merges)) if merges else 0.0

    return {
        "suite_size": K,
        "reduction_ratio": 1.0 - K / N,
        "marginal_preserved": marginal_preserved,
        "cross_verdict_merge_rate": cross_rate,
        "n_merges": len(merges),
        "n_cross_merges": cross_merge,
    }


def main():
    raw = load_raw_cases()
    N = len(raw)
    if N == 0:
        print("ERROR: no raw cases loaded.")
        return

    verdicts = Counter(c["verdict"] for c in raw)
    print(f"=== Input: {N} raw cases ===")
    print(f"Verdict dist: {dict(verdicts)}")
    print()

    rows = []
    print(f"{'θ':>6} │ {'kept':>5} │ {'red%':>6} │ {'marginal%':>10} │ {'cross_merge':>12} │ {'n_merges':>9}")
    print("─" * 72)

    for theta in THETAS:
        kept, merges = greedy_dedup(raw, theta)
        m = metrics(raw, kept, merges)
        rows.append((theta, m))
        flag = "  ← 0.15 (default)" if abs(theta - 0.15) < 1e-9 else ""
        print(f"{theta:>6.2f} │ {m['suite_size']:>5d} │ "
              f"{m['reduction_ratio']*100:>5.1f}% │ "
              f"{m['marginal_preserved']*100:>9.1f}% │ "
              f"{m['cross_verdict_merge_rate']*100:>9.1f}% (n={m['n_cross_merges']}) │ "
              f"{m['n_merges']:>9d}{flag}")

    # CSV 저장
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8") as fp:
        fp.write("theta,suite_size,reduction_ratio,marginal_preserved,cross_verdict_merge_rate,n_merges,n_cross_merges\n")
        for theta, m in rows:
            fp.write(f"{theta},{m['suite_size']},{m['reduction_ratio']:.4f},"
                     f"{m['marginal_preserved']:.4f},{m['cross_verdict_merge_rate']:.4f},"
                     f"{m['n_merges']},{m['n_cross_merges']}\n")
    print(f"\n💾 saved: {OUT_CSV}")

    # 평탄 영역 분석: θ ∈ [0.10, 0.15] 의 suite_size 변동
    flat_region = [m for theta, m in rows if 0.10 <= theta <= 0.15]
    if len(flat_region) >= 2:
        sizes = [m["suite_size"] for m in flat_region]
        margs = [m["marginal_preserved"] for m in flat_region]
        print()
        print("▎ 평탄 영역 분석 (θ ∈ [0.10, 0.15])")
        print(f"  Suite size:           {sizes}  (변동 {max(sizes)-min(sizes)})")
        print(f"  Marginal preserved:   {[f'{x:.2f}' for x in margs]}")
        if max(sizes) - min(sizes) <= max(2, int(0.1 * max(sizes))):
            print("  → 평탄 영역 확인 ✓ (0.15는 안정 구간 중심)")
        else:
            print("  → 평탄 영역 아님 — 추가 검토 필요")


if __name__ == "__main__":
    main()
