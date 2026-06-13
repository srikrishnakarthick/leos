# LEOS — Light Environment Observatory for the Solar System

> ⚠️ **Status: Early development — Phase 0 (Infrastructure)**

A multi-language scientific framework for high-fidelity solar illumination,
spectral irradiance, and energy availability computations across every body
in the Solar System.

---

## What LEOS does

- Computes solar irradiance and spectral power at any surface, orbital, or flyby location
- Models atmospheric radiative transfer for Earth, Mars, Moon, and beyond
- Propagates uncertainty through every output — results are always `I ± σ`
- Accepts SPICE kernels to compute irradiance along real mission trajectories
- Models solar panel power generation anywhere in the Solar System
- Finds solar-optimal landing sites and trajectories via inverse optimization

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Infrastructure — build system, data types, SPICE geometry engine | 🔄 In progress |
| 1 | Validated core irradiance — Earth, Mars, Moon | ⏳ Planned |
| 2 | Trajectory support and solar panel power modeling | ⏳ Planned |
| 3 | Full Solar System expansion and uncertainty quantification | ⏳ Planned |
| 4 | ML surrogates and inverse design | ⏳ Planned |
| 5 | Web platform and REST API | ⏳ Planned |

---

## Stack

- **Python** — user-facing API, SpiceyPy interface, astropy.units throughout
- **Julia** — high-performance parameter sweeps, Monte Carlo UQ, optimization
- **C/C++** — SPICE kernels, radiative transfer core

---

## Installation

Coming soon.

---

## License

MIT
