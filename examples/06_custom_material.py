"""Custom material: define an OrthotropicMaterial not in the library.

The preset library (``MaterialLibrary().list_names()``) covers common
prepregs; for anything else, construct ``OrthotropicMaterial`` directly
with your elastic constants and strength allowables (moduli and
strengths in MPa) and pass it to ``AnalysisConfig``.

Expected runtime: ~1 s (analytical-only).
Expected output:  the library names, then knockdown/strength for the
                  custom S-glass/epoxy compared against IM7/8552.
"""

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial

print("Preset materials:", ", ".join(MaterialLibrary().list_names()))

# Representative S-glass/epoxy UD ply (values for illustration only —
# use your own qualified allowables for real work).
s_glass = OrthotropicMaterial(
    name="S-glass/epoxy (custom)",
    E1=52000.0, E2=15000.0, E3=15000.0,
    G12=4700.0, G13=4700.0, G23=4300.0,
    nu12=0.28, nu13=0.28, nu23=0.40,
    Xt=1700.0, Xc=1000.0,
    Yt=60.0, Yc=140.0,
    Zt=60.0, Zc=140.0,
    S12=70.0, S13=70.0, S23=50.0,
)

for mat in (s_glass, MaterialLibrary().get("IM7_8552")):
    config = AnalysisConfig(
        amplitude=0.366, wavelength=16.0, width=12.0,
        morphology="stack", loading="compression",
        material=mat,
    )
    r = WrinkleAnalysis(config).run(analytical_only=True)
    print(
        f"{mat.name:<28} knockdown={r.analytical_knockdown:.4f} "
        f"strength={r.analytical_strength_MPa:.1f} MPa"
    )
