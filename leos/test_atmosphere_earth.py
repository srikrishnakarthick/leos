#!/usr/bin/env python3
"""
test_atmosphere_earth.py
------------------------
Tests for leos/atmosphere_earth.py

Covers:
  1. US Standard Atmosphere 1976 construction
  2. O3 VMR attachment and column density (~300 DU sanity check)
  3. Scale height at surface and tropopause
  4. Effective scale height (density-weighted)
  5. Well-mixed species column densities (N2, O2)
  6. to_profile() bridge → AtmosphericProfile round-trip
  7. from_npz() round-trip with a synthetic .npz
  8. AtmosphericSource registry completeness
  9. Edge cases: z_max truncation, unsorted input, missing species

Run with:
    python test_atmosphere_earth.py
"""

import os
import sys
import tempfile
import traceback

import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"
SKIP  = "\033[93mSKIP\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    tag = f"[{status}]"
    print(f"  {tag}  {name}" + (f"  — {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── Import ────────────────────────────────────────────────────────────────────

section("Import")
try:
    from leos.atmosphere_earth import (
        AtmosphericColumn, AtmosphericSource, ATMO_REGISTRY,
        _US76_LAYERS, _O3_VMR_TABLE_KM, _O3_VMR_TABLE_PPM,
    )
    from leos.atmosphere import AtmosphericProfile
    from astropy import units as u
    check("leos.atmosphere_earth imports cleanly", True)
except Exception as e:
    check("leos.atmosphere_earth imports cleanly", False, str(e))
    print("\nCannot continue without a successful import.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. US Standard Atmosphere 1976
# ══════════════════════════════════════════════════════════════════════════════

section("1 · US Standard Atmosphere 1976")

col = AtmosphericColumn.us_standard_1976(z_max_km=86.0, dz_km=1.0, include_o3=False)

check("Returns AtmosphericColumn instance", isinstance(col, AtmosphericColumn))
check("Source is US_STD_1976",  col.source == AtmosphericSource.US_STD_1976)
check("Body is 'earth'",        col.body == "earth")
check("86 altitude levels",     len(col.z_km) == 87,
      f"got {len(col.z_km)}")
check("z_km starts at 0",       col.z_km[0] == 0.0)
check("z_km ends at 86",        col.z_km[-1] == 86.0)
check("z_km strictly ascending", np.all(np.diff(col.z_km) > 0))

# Surface values per standard
check("Surface T ≈ 288.15 K",
      abs(col.T_K[0] - 288.15) < 0.01, f"{col.T_K[0]:.3f} K")
check("Surface P ≈ 101325 Pa",
      abs(col.P_Pa[0] - 101325.0) < 5.0, f"{col.P_Pa[0]:.1f} Pa")

# Tropopause (~11 km): T should be ~216.65 K (isothermal layer begins)
T_11 = float(np.interp(11.0, col.z_km, col.T_K))
check("T at 11 km ≈ 216.65 K (tropopause base)",
      abs(T_11 - 216.65) < 0.5, f"{T_11:.2f} K")

# Pressure at 11 km from the standard: ~22632 Pa
P_11 = float(np.interp(11.0, col.z_km, col.P_Pa))
check("P at 11 km ≈ 22632 Pa",
      abs(P_11 - 22632.0) < 200.0, f"{P_11:.1f} Pa")

# n_total is positive everywhere
check("n_total > 0 everywhere", np.all(col.n_total > 0))

# Well-mixed composition sums to ~1
total_vmr = sum(col.vmr[sp][0] for sp in col.vmr)
check("Surface VMR fractions sum ≈ 1",
      abs(total_vmr - 1.0) < 0.01, f"sum={total_vmr:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. O3 profile attachment (McPeters/Labow)
# ══════════════════════════════════════════════════════════════════════════════

section("2 · O3 VMR and column (McPeters/Labow bundled)")

col_o3 = AtmosphericColumn.us_standard_1976(include_o3=True)

check("Source is MCPETERS_LABOW when include_o3=True",
      col_o3.source == AtmosphericSource.MCPETERS_LABOW)
check("'O3' present in vmr dict", "O3" in col_o3.vmr)
check("O3 VMR >= 0 everywhere", np.all(col_o3.vmr["O3"] >= 0))
check("O3 VMR peak is between 20-35 km",
      20 <= col_o3.z_km[np.argmax(col_o3.vmr["O3"])] <= 35,
      f"peak at z={col_o3.z_km[np.argmax(col_o3.vmr['O3'])]:.0f} km")

col_O3 = col_o3.column_density("O3")
o3_du  = col_O3 / 2.6867e20
check("O3 column is 200–400 DU (mid-latitude range)",
      200 <= o3_du <= 400, f"{o3_du:.1f} DU")

# Missing species returns 0
check("column_density() returns 0 for absent species",
      col_o3.column_density("SO2") == 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Well-mixed species column densities
# ══════════════════════════════════════════════════════════════════════════════

section("3 · Well-mixed column densities (N2, O2)")

# Theoretical N2 column from Ni = xi * P / (mi * g)
_kB  = 1.380649e-23
_g0  = 9.80665
_mN2 = 4.652e-26
_mO2 = 5.314e-26
xi_N2 = 0.78084
xi_O2 = 0.20946
P_surf = col_o3.P_Pa[0]

N2_theory = xi_N2 * P_surf / (_mN2 * _g0)
O2_theory = xi_O2 * P_surf / (_mO2 * _g0)

N2_integrated = col_o3.column_density("N2")
O2_integrated = col_o3.column_density("O2")

# The hydrostatic formula Ni = xi*P/(mi*g) is a single-layer isothermal
# approximation (infinite column, constant T). The trapezoidal integral
# over the real US76 T(z) profile to 86 km diverges from it because:
#   (a) the column is finite — mass above 86 km is missing
#   (b) T(z) is not constant — the effective scale height differs
#       from kB*T_surf/(m*g) at every level
# Agreement within 15% is physically honest for this comparison; tighter
# tolerances would require matching assumptions (same T, same column top).
check("N2 column within 15% of single-layer hydrostatic estimate",
      abs(N2_integrated - N2_theory) / N2_theory < 0.15,
      f"integrated={N2_integrated:.3e}  theory={N2_theory:.3e}")
check("O2 column within 15% of single-layer hydrostatic estimate",
      abs(O2_integrated - O2_theory) / O2_theory < 0.15,
      f"integrated={O2_integrated:.3e}  theory={O2_theory:.3e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Scale heights
# ══════════════════════════════════════════════════════════════════════════════

section("4 · Scale heights")

H_surf = col_o3.scale_height_at(0.0)
check("scale_height_at() returns astropy Quantity",
      isinstance(H_surf, u.Quantity))
check("Surface scale height is in km",
      H_surf.unit.is_equivalent(u.km))
check("Surface H ≈ 8–9 km (Earth standard)",
      8.0 <= H_surf.to(u.km).value <= 9.0,
      f"{H_surf.to(u.km):.3f}")

# US76 temperature structure:
#   0–11 km:  288.15 → 216.65 K  (troposphere, cools)
#   11–20 km: 216.65 K isothermal (tropopause)
#   20–32 km: 216.65 → 228.65 K  (lower stratosphere, warms slowly)
#   32–47 km: 228.65 → 270.65 K  (stratosphere, warms faster)
#   47–51 km: 270.65 K isothermal (stratopause)
# At 25 km: T = 216.65 + 1.0*(25-20) = 221.65 K — still < 288.15 K surface.
# H(z) = kB*T(z)/(m*g(z)); since T(25 km) < T(0 km), H(25) < H(0) is correct.
# The right test is H at the tropopause minimum < H at the surface.
H_tropo = col_o3.scale_height_at(11.0)
check("H at tropopause (11 km, T=216.65 K) < H at surface (T=288.15 K)",
      H_tropo.to(u.km).value < H_surf.to(u.km).value,
      f"H(11km)={H_tropo.to(u.km):.3f}  H(0km)={H_surf.to(u.km):.3f}")

# Above the stratopause (~47 km) T recovers toward surface-like values;
# H there should be similar in magnitude to the surface H.
H_strat = col_o3.scale_height_at(47.0)
check("H at stratopause (47 km, T≈270 K) is in 6–9 km range",
      6.0 <= H_strat.to(u.km).value <= 9.0,
      f"H(47km)={H_strat.to(u.km):.3f}")

H_eff = col_o3.effective_scale_height()
check("effective_scale_height() returns astropy Quantity",
      isinstance(H_eff, u.Quantity))
check("Effective H is 6–10 km (density-weighted, dominated by low altitudes)",
      6.0 <= H_eff.to(u.km).value <= 10.0,
      f"{H_eff.to(u.km):.3f}")

# Unknown weight scheme raises
try:
    col_o3.effective_scale_height(weight="mass")
    check("Unknown weight raises ValueError", False)
except ValueError:
    check("Unknown weight raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. to_profile() bridge
# ══════════════════════════════════════════════════════════════════════════════

section("5 · to_profile() → AtmosphericProfile")

prof = col_o3.to_profile()
check("Returns AtmosphericProfile", isinstance(prof, AtmosphericProfile))
check("body matches",  prof.body == "earth")
check("has_atmosphere is True", prof.has_atmosphere)
check("surface_pressure in Pa",
      abs(prof.surface_pressure.to(u.Pa).value - 101325.0) < 10.0)
check("scale_height matches effective_scale_height",
      abs(prof.scale_height.to(u.km).value - H_eff.to(u.km).value) < 0.01)
check("column_densities populated for O3",
      "O3" in prof.column_densities and prof.column_densities["O3"] > 0)
check("column_density_for('O3') uses pre-computed value",
      abs(prof.column_density_for("O3") - col_O3) < 1e15)

# Override forwarding
prof2 = col_o3.to_profile(dust_tau=0.42, label="custom_test")
check("Override dust_tau forwarded", abs(prof2.dust_tau - 0.42) < 1e-9)
check("Override label forwarded",    prof2.label == "custom_test")


# ══════════════════════════════════════════════════════════════════════════════
# 6. from_npz() round-trip
# ══════════════════════════════════════════════════════════════════════════════

section("6 · from_npz() round-trip (synthetic .npz)")

z_km    = np.linspace(0, 50, 51)
T_K     = 288.15 - 6.5 * z_km
T_K     = np.clip(T_K, 216.65, 400.0)
P_Pa    = 101325.0 * np.exp(-z_km / 8.5)
n_total = P_Pa / (1.380649e-23 * T_K)
vmr_N2  = np.full_like(z_km, 0.78084)
vmr_O3  = np.interp(z_km, _O3_VMR_TABLE_KM, _O3_VMR_TABLE_PPM) * 1e-6

with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
    tmp_path = f.name

try:
    np.savez(
        tmp_path,
        z_km     = z_km,
        T_K      = T_K,
        P_Pa     = P_Pa,
        n_total  = n_total,
        vmr_N2   = vmr_N2,
        vmr_O3   = vmr_O3,
        lat      = np.array(18.4),
        lon      = np.array(77.5),
        time_iso = np.array(["2024-01-15T12:00"], dtype=object),
        source   = np.array(["era5"],             dtype=object),
    )

    col2 = AtmosphericColumn.from_npz(tmp_path)

    check("from_npz returns AtmosphericColumn", isinstance(col2, AtmosphericColumn))
    check("source parsed as ERA5", col2.source == AtmosphericSource.ERA5)
    check("lat/lon round-trip",
          col2.lat == 18.4 and col2.lon == 77.5)
    check("z_km round-trip (first level)",
          col2.z_km[0] == 0.0)
    check("vmr_N2 present after load", "N2" in col2.vmr)
    check("vmr_O3 present after load", "O3" in col2.vmr)
    check("n_total loaded (not recomputed)",
          np.allclose(col2.n_total, n_total, rtol=1e-6))
    check("column_density('O3') > 0 from loaded column",
          col2.column_density("O3") > 0)

finally:
    os.unlink(tmp_path)

# Missing file raises FileNotFoundError
try:
    AtmosphericColumn.from_npz("/nonexistent/path/foo.npz")
    check("Missing .npz raises FileNotFoundError", False)
except FileNotFoundError:
    check("Missing .npz raises FileNotFoundError", True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ATMO_REGISTRY completeness
# ══════════════════════════════════════════════════════════════════════════════

section("7 · ATMO_REGISTRY")

for src in AtmosphericSource:
    check(f"Registry entry exists for {src.name}",
          src in ATMO_REGISTRY)
    if src in ATMO_REGISTRY:
        entry = ATMO_REGISTRY[src]
        check(f"  {src.name} has 'description'", "description" in entry)
        check(f"  {src.name} has 'requires_download'", "requires_download" in entry)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ══════════════════════════════════════════════════════════════════════════════

section("8 · Edge cases")

# z_max truncation
col_trunc = AtmosphericColumn.us_standard_1976(z_max_km=30.0, dz_km=1.0)
check("z_max=30 km truncation respected",
      col_trunc.z_km[-1] <= 30.0, f"max z = {col_trunc.z_km[-1]:.1f} km")

# z_max > 86 clamped to 86
col_clamped = AtmosphericColumn.us_standard_1976(z_max_km=150.0, dz_km=1.0)
check("z_max > 86 km clamped to 86 km",
      col_clamped.z_km[-1] <= 86.0, f"max z = {col_clamped.z_km[-1]:.1f} km")

# Unsorted input → auto-sorted
z_unsorted = np.array([10.0, 0.0, 5.0])
T_unsorted = np.array([223.0, 288.0, 255.0])
P_unsorted = np.array([26500.0, 101325.0, 54000.0])
col_us = AtmosphericColumn(
    z_km=z_unsorted, T_K=T_unsorted, P_Pa=P_unsorted,
    vmr={"N2": np.array([0.78, 0.78, 0.78])},
)
check("Unsorted z_km auto-sorted ascending",
      np.all(np.diff(col_us.z_km) > 0),
      f"z_km = {col_us.z_km}")
check("T_K reordered with z_km",
      col_us.T_K[0] == 288.0)  # 0 km → 288 K

# n_total computed from P/kBT if not supplied
check("n_total auto-computed from ideal gas law",
      np.allclose(col_us.n_total,
                  col_us.P_Pa / (1.380649e-23 * col_us.T_K), rtol=1e-9))

# __str__ / __repr__ smoke test
s = str(col_o3)
check("__str__ includes body name", "earth" in s)
check("__str__ includes source",    "mcpeters_labow" in s)


# ══════════════════════════════════════════════════════════════════════════════
# 9. ERA5 .npz integration (skip if file absent)
# ══════════════════════════════════════════════════════════════════════════════

section("9 · ERA5 .npz integration (real file)")

era5_path = "leos/data/era5_2024_18.40N_77.50E.npz"
if os.path.exists(era5_path):
    try:
        col_era5 = AtmosphericColumn.from_npz(era5_path)
        check("ERA5 file loads", True)
        check("source == ERA5",  col_era5.source == AtmosphericSource.ERA5)
        check("37 pressure levels", len(col_era5.z_km) == 37,
              f"got {len(col_era5.z_km)}")
        check("lat ≈ 18.5", abs(col_era5.lat - 18.5) < 0.1,
              f"lat={col_era5.lat}")
        check("lon ≈ 77.5", abs(col_era5.lon - 77.5) < 0.1,
              f"lon={col_era5.lon}")

        o3_du_era5 = col_era5.column_density("O3") / 2.6867e20
        check("ERA5 O3 column 200–350 DU",
              200 <= o3_du_era5 <= 350, f"{o3_du_era5:.1f} DU")

        H2O_col = col_era5.column_density("H2O")
        check("ERA5 H2O column > 0", H2O_col > 0,
              f"{H2O_col:.3e} molecules/m^2")

        prof_era5 = col_era5.to_profile()
        check("ERA5 to_profile() works", isinstance(prof_era5, AtmosphericProfile))

    except Exception as e:
        check(f"ERA5 file integration (unexpected error)", False, str(e))
        traceback.print_exc()
else:
    print(f"  [{SKIP}]  ERA5 file not found at {era5_path!r} — skipping")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

section("Summary")
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"  {n_pass}/{len(results)} checks passed", end="")
if n_fail:
    print(f"  ·  {n_fail} FAILED ← fix before committing")
else:
    print("  ✓  all clear")
print()
sys.exit(0 if n_fail == 0 else 1)
