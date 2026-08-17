"""Acceptance limit: the largest wrinkle amplitude that still passes.

Inverts the forward model. Instead of "given this wrinkle, what is the
knockdown?", this answers the question an NCR disposition is written
around: "given this allowable, what is the largest wrinkle we can
accept?" ``find_critical_value`` scans the resolved amplitude range,
brackets the single crossing of ``knockdown - target``, root-finds it,
and then backs the answer off to the safe side — so the value it returns
satisfies the criterion under a real forward evaluation, not merely to
within the root tolerance. The script re-runs the forward model at each
answer to show that.

Expected runtime: ~5 s (analytical-only).
Expected output:  the scan table and the acceptance limit for KD >= 0.85
                  and KD >= 0.60, each verified by a forward re-run, plus
                  ``09_acceptance_limit.png``.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove to use an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.goalseek import find_critical_value

base = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)

targets = (0.85, 0.60)
answers = {}

for target in targets:
    result = find_critical_value(base, target_knockdown=target)
    print("=" * 66)
    print(f"Target: knockdown >= {target:.2f}")
    print("=" * 66)
    print(result.summary())
    print()

    if result.status != "converged":
        # A refusal is a real outcome, not a crash: the message names the
        # measurement behind it and the knob that has to move.
        print(f"No acceptance limit returned ({result.status}).")
        print(result.message)
        print()
        continue

    # The guarantee, demonstrated: re-run the forward model at the
    # returned config and check the criterion directly.
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    achieved = forward.analytical_knockdown
    print(
        f"Forward re-run at A = {result.critical_value:.6f} mm: "
        f"knockdown {achieved:.6f} >= {target:.6f} -> "
        f"{'PASS' if achieved >= target else 'FAIL'}"
    )
    print(
        f"  (raw brentq root {result.critical_value_root:.6f} mm, "
        f"backed off to the safe side)"
    )
    print(f"  predicted strength {forward.analytical_strength_MPa:.1f} MPa")
    print()
    answers[target] = result.critical_value

# The inverse of example 02's figure: the same knockdown curve, read
# horizontally from the allowable to the acceptance limit.
amplitudes = np.linspace(0.0, 0.5, 41)
curve = [
    WrinkleAnalysis(
        AnalysisConfig(
            amplitude=float(a), wavelength=16.0, width=12.0,
            morphology="stack", loading="compression",
        )
    ).run(analytical_only=True).analytical_knockdown
    for a in amplitudes
]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(amplitudes, curve, "-", color="#1f77b4", label="knockdown")
colors = {0.85: "#d62728", 0.60: "#2ca02c"}
for target, amplitude in answers.items():
    ax.axhline(target, color=colors[target], ls="--", lw=1.0)
    ax.axvline(amplitude, color=colors[target], ls=":", lw=1.0)
    ax.plot([amplitude], [target], "o", color=colors[target])
    ax.annotate(
        f"KD >= {target:.2f}  ->  A <= {amplitude:.4f} mm",
        xy=(amplitude, target), xytext=(amplitude + 0.03, target + 0.03),
        color=colors[target], fontsize=9,
    )

ax.set_xlabel("wrinkle half-amplitude A (mm)")
ax.set_ylabel("strength knockdown")
ax.set_title("Acceptance limit: largest amplitude meeting each allowable")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("09_acceptance_limit.png", dpi=150)
print("Saved: 09_acceptance_limit.png")
