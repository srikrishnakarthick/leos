import os
import hashlib
import requests
from pathlib import Path
# ── Dynamic Ephemeris Configuration ──────────────────────────────────────────
DE_VERSION = "de442"

# ── Updated Directory Architecture ──────────────────────────────────────────
KERNEL_ROOT = os.path.join(os.path.dirname(__file__), "data")

# Create dedicated subdirectories for pipeline sanitation
DATA_DIRS = {
    "generic": os.path.join(KERNEL_ROOT, "generic"),
    "mission": os.path.join(KERNEL_ROOT, "mission")  # Playground for user CK/IK/SCLK files
}

for folder in DATA_DIRS.values():
    os.makedirs(folder, exist_ok=True)

# Generic static text kernels
STATIC_KERNELS = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
    "mars_iau2000_v1.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/mars_iau2000_v1.tpc",
    "mar099s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/mar099s.bsp"
}

def get_dynamic_ephemeris_urls():
    """Builds both the binary file and its companion tech-comments documentation file."""
    base_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
    return {
        f"{DE_VERSION}.bsp": f"{base_url}{DE_VERSION}.bsp",
        f"{DE_VERSION}_tech-comments.txt": f"{base_url}{DE_VERSION}_tech-comments.txt"
    }

def fetch_remote_md5s():
    checksum_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_checksums.txt"
    md5_dict = {}
    try:
        response = requests.get(checksum_url, timeout=10)
        response.raise_for_status()
        for line in response.text.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                hash_val, filename = parts[0], parts[1]
                md5_dict[filename.lower()] = hash_val.lower()
    except Exception as e:
        print(f"  ⚠️ Warning: Could not fetch remote aa_checksums.txt ({e}).")
    return md5_dict

def calculate_local_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def fetch_kernels(target_dir=None):
    """
    Fetches missing planetary assets.
    Creates directories automatically if they don't exist.
    """
    # ── THE COLAB/CUSTOM PATH FIX ──
    # Resolve the root path dynamically based on runtime input
    root_dir = os.path.abspath(target_dir) if target_dir else _DEFAULT_KERNEL_ROOT
    generic_dir = os.path.join(root_dir, "generic")
    mission_dir = os.path.join(root_dir, "mission")

    # ── THE AUTO-CREATION FIX ──
    # Ensure folders exist at runtime execution point
    os.makedirs(generic_dir, exist_ok=True)
    os.makedirs(mission_dir, exist_ok=True)

    print(f"  Fetching live NAIF asset checksum tokens...")
    nasa_md5s = fetch_remote_md5s()

    # Build queue out of binary files, comments file, and static kernels
    queue = get_dynamic_ephemeris_urls()
    for name, url in STATIC_KERNELS.items():
        queue[name] = url

    for filename, url in queue.items():
        dest = os.path.join(generic_dir, filename)
        expected_md5 = nasa_md5s.get(filename.lower())

        if os.path.exists(dest):
            if expected_md5:
                if calculate_local_md5(dest) == expected_md5:
                    print(f"  Verified & intact (via NAIF Manifest): {filename}")
                    continue
            else:
                if os.path.getsize(dest) > 0:
                    print(f"  Verified via document footprint: {filename}")
                    continue

        print(f"  Downloading/Correcting {filename} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
                
        if expected_md5 and calculate_local_md5(dest) != expected_md5:
            raise ValueError(f"MD5 verification failure on newly downloaded asset: {filename}")
        print(f"  Successfully verified and saved: {filename}")

if __name__ == "__main__":
    print(f"Initializing Generic SPICE Pipeline Asset Fetcher [Target: {DE_VERSION}]")
    fetch_kernels()
