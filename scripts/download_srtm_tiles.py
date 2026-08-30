#!/usr/bin/env python3
"""
Fetch specific SRTM 1-arc-second elevation tiles into the local offline
cache (~/.depthwizard-x/srtm_cache/ by default), for use by
geospatial/srtm_cache.py as an offline reference DEM source.

Never downloads silently: prints tile name, source, license, and target
path before fetching, and NEVER invents/interpolates a missing tile --
if a tile can't be fetched, it prints manual-download instructions and
exits non-zero rather than proceeding with partial data.

USAGE:
    python scripts/download_srtm_tiles.py N28E077 N28E078
    python scripts/download_srtm_tiles.py --for-bounds 76.8 28.4 77.3 28.7

The second form computes the exact tile names needed to cover a WGS84
bounding box (min_lon min_lat max_lon max_lat) using the same tile-naming
logic the pipeline itself uses (geospatial.srtm_cache.required_tiles_for_bounds),
so "which tiles do I need for my scene" and "which tiles will the
pipeline look for" can never silently disagree.

SOURCE: OpenTopography's public SRTMGL1 (30m) mirror is used here as one
example source with a documented, redistributable license. Swap SOURCE_URL_TEMPLATE
below to your organization's preferred authoritative mirror (e.g. NASA
Earthdata, which requires an account) before relying on this for a
production deployment.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from depthwizard.geospatial.srtm_cache import DEFAULT_CACHE_DIR, required_tiles_for_bounds  # noqa: E402

SOURCE_LICENSE = "USGS/NASA SRTM -- public domain (US Government work)"
SOURCE_URL_TEMPLATE = "https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/{tile}.tif"


def download_tile(tile_name: str, cache_dir: Path) -> bool:
    target = cache_dir / f"{tile_name}.tif"
    url = SOURCE_URL_TEMPLATE.format(tile=tile_name)

    print(f"Tile: {tile_name}")
    print(f"  License: {SOURCE_LICENSE}")
    print(f"  Source: {url}")
    print(f"  Target: {target}")

    if target.exists():
        print("  Already cached, skipping.")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
        print(f"  -> done ({target.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        target.unlink(missing_ok=True)
        print(f"  FAILED: {e}")
        print(
            "  Automatic download failed (this mirror may not have every tile, "
            "or network access may be restricted in this environment). Manual "
            "download instructions:\n"
            f"    1. Obtain a GeoTIFF DEM covering tile {tile_name} from a "
            f"trusted source (e.g. https://portal.opentopography.org or NASA "
            f"Earthdata: https://earthexplorer.usgs.gov)\n"
            f"    2. Reproject/save it as EPSG:4326 GeoTIFF named exactly "
            f"'{tile_name}.tif'\n"
            f"    3. Place it at: {target}\n"
        )
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiles", nargs="*", help="Explicit tile names, e.g. N28E077")
    parser.add_argument(
        "--for-bounds", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Compute required tile names for a WGS84 bounding box instead of naming them explicitly.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    if args.for_bounds:
        tiles = required_tiles_for_bounds(*args.for_bounds)
        print(f"Bounding box requires {len(tiles)} tile(s): {tiles}\n")
    else:
        tiles = args.tiles

    if not tiles:
        parser.error("Provide explicit tile names, or --for-bounds MIN_LON MIN_LAT MAX_LON MAX_LAT")

    results = [download_tile(t, args.cache_dir) for t in tiles]
    if not all(results):
        sys.exit(1)
    print(f"All {len(tiles)} tile(s) ready in {args.cache_dir}")


if __name__ == "__main__":
    main()
