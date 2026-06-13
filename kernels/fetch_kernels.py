"""
kernels/fetch_kernels.py
------------------------
Downloads the minimum SPICE kernels required for LEOS Phase 1
(Earth, Moon, Mars) from NASA NAIF and saves them to kernels/data/.

Run once:
    python kernels/fetch_kernels.py

Kernel types:
    LSK  — leap seconds kernel (time conversion)
    PCK  — planetary constants (radii, rotation)
    SPK  — solar system ephemeris (positions)
"""

import os
import requests

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(KERNEL_DIR, exist_ok=True)

KERNELS = {
    # Leap seconds — required for all time conversions
    "naif0012.tls": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
    ),
    # Planetary constants — radii, rotation, gravity
    "pck00011.tpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc"
    ),
    # Solar system ephemeris — positions of planets and Moon
    "de440.bsp": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"
    ),
    # Mars orientation — required for IAU_MARS body-fixed frame
    "mars_iau2000_v1.tpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/mars_iau2000_v1.tpc"
    )
}


def fetch_kernels():
    for filename, url in KERNELS.items():
        dest = os.path.join(KERNEL_DIR, filename)
        if os.path.exists(dest):
            print(f"  already exists: {filename}")
            continue
        print(f"  downloading {filename} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  saved {filename} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("Fetching SPICE kernels...")
    fetch_kernels()
    print("Done. Kernels saved to kernels/data/")
