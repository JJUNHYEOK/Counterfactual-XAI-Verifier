"""
Phase 0 — L1 < theta 임계값의 물리적 의미 분해.

정규화 좌표계:
    x_fog   = fog   / 100      [%]
    x_illum = illum / 15000    [lx]
    x_noise = noise            [0..1]

L1 거리 = |Δx_fog| + |Δx_illum| + |Δx_noise|
임계값  θ = 0.15  (정규화 공간) — sensitivity analysis 로 검증된 default.

목표: 사용자가 직관적으로 이해할 수 있도록
      "L1 < theta 인 두 케이스는 어느 정도 가까운가?"
      를 운용 단위(% / lx / noise)로 표현.
"""
from __future__ import annotations

THETA = 0.15
FOG_RANGE = 100.0       # %
ILLUM_RANGE = 15000.0   # lx
NOISE_RANGE = 1.0       # [0..1]

def axis_solo(theta: float) -> dict:
    """다른 두 축이 0 일 때 단일 축이 가질 수 있는 최대 변화."""
    return {
        "fog_pp":    theta * FOG_RANGE,     # percentage points
        "illum_lx":  theta * ILLUM_RANGE,   # lux
        "noise_abs": theta * NOISE_RANGE,   # absolute units
    }

def axis_uniform(theta: float) -> dict:
    """세 축이 균등하게 변할 때 각 축당 변화."""
    each = theta / 3.0
    return {
        "fog_pp":    each * FOG_RANGE,
        "illum_lx":  each * ILLUM_RANGE,
        "noise_abs": each * NOISE_RANGE,
    }

def print_table():
    solo = axis_solo(THETA)
    uni  = axis_uniform(THETA)

    print(f"=== L1 < {THETA} 의 물리적 의미 ===\n")
    print(f"정규화: fog/100, illum/15000, noise/1")
    print(f"L1(a,b) = |Δfog/100| + |Δillum/15000| + |Δnoise|\n")

    print("┌──────────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ 시나리오                     │ Δfog (%p)    │ Δillum (lx)  │ Δnoise       │")
    print("├──────────────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ 단일 축 (나머지=0)           │ ±{solo['fog_pp']:>6.2f}      │ ±{solo['illum_lx']:>6.0f}       │ ±{solo['noise_abs']:>6.3f}      │")
    print(f"│ 균등 혼합 (3축 동시)         │ ±{uni['fog_pp']:>6.2f}      │ ±{uni['illum_lx']:>6.0f}       │ ±{uni['noise_abs']:>6.3f}      │")
    print("└──────────────────────────────┴──────────────┴──────────────┴──────────────┘")

    fog_pp = solo['fog_pp']
    illum_lx = solo['illum_lx']
    noi_abs = solo['noise_abs']
    print()
    print("▎ 직관적 해석")
    print(f"  · 안개율 차이 {fog_pp:.0f}%p 이내    (예: 35% ↔ {35+int(fog_pp)}%)")
    print(f"  · 조도 차이 {illum_lx:.0f} lx 이내     (예: 6000 ↔ {6000+int(illum_lx)})")
    print(f"  · 노이즈 차이 {noi_abs:.2f} 이내        (예: 0.10 ↔ {0.10+noi_abs:.2f})")
    print(f"  · 세 축이 동시에 작게 변할 때: 각각 ±{uni['fog_pp']:.1f}%p / ±{uni['illum_lx']:.0f}lx / ±{uni['noise_abs']:.3f}")
    print()
    print("▎ 결론")
    print(f"  L1 < {THETA} 는 운용 관점에서 '동일 임무 envelope 내 변동'으로 해석 가능.")
    print("  → 사용자 직관적 '같은 케이스' 판정 기준과 정합.")


if __name__ == "__main__":
    print_table()
