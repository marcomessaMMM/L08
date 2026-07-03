#!/usr/bin/env python3
"""
fc.py  —  Standalone Finding Chart Generator with Tkinter GUI
=============================================================

Creates astronomical finding charts by querying public sky-survey servers.
No dependency on the SOXSScheduler Utils package: everything needed is
included directly in this file.

Supported surveys and their photometric filters
-----------------------------------------------
  Legacy Survey   : g, r, i, z
  Pan-STARRS      : g, r, i, z, y
  SkyMapper       : g, r, i, z
  2MASS           : J, H, K
  UKIDSS          : J, H, K
  GAIA (SkyView)  : G

Usage
-----
    python fc.py

Requirements
------------
    pip install pillow astropy matplotlib requests pandas numpy
"""
import io
import os
import warnings
import urllib.parse

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive backend: renders to file/memory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

import threading
import requests
from io import BytesIO, StringIO

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.visualization.wcsaxes import add_scalebar

from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Matplotlib font settings
# ---------------------------------------------------------------------------
rcParams["font.family"] = "serif"
rcParams["font.serif"]  = ["DejaVu Serif"]

# ---------------------------------------------------------------------------
# Survey catalogue definitions
#
# Each entry stores:
#   label        – human-readable name shown in the GUI
#   pix_scale    – native pixel scale in arcsec/pixel
#   size_scaling – factor to convert arcmin → request unit
#                  (pixels for Legacy/PS1, degrees for the rest)
#   filters      – list of filters available for this survey
# ---------------------------------------------------------------------------
SURVEYS = {
    "legacy": {
        "label":        "Legacy Survey",
        "pix_scale":    0.262,
        "size_scaling": 60 / 0.252,   # arcmin → pixels  (≈ 1 pix = 0.252 arcsec)
        "filters":      ["g", "r", "i", "z"],
    },
    "ps1": {
        "label":        "Pan-STARRS",
        "pix_scale":    0.25,
        "size_scaling": 60 / 0.25,    # arcmin → pixels  (1 pix = 0.25 arcsec)
        "filters":      ["g", "r", "i", "z", "y"],
    },
    "skymapper": {
        "label":        "SkyMapper",
        "pix_scale":    0.50,
        "size_scaling": 1 / 60,       # arcmin → degrees
        "filters":      ["g", "r", "i", "z"],
    },
    "2mass": {
        "label":        "2MASS",
        "pix_scale":    2.00,
        "size_scaling": 1 / 60,       # arcmin → degrees
        "filters":      ["J", "H", "K"],
    },
    "ukidss": {
        "label":        "UKIDSS",
        "pix_scale":    0.40,
        "size_scaling": 1 / 60,       # arcmin → degrees
        "filters":      ["J", "H", "K"],
    },
}

# Reverse map: display label → internal key  (e.g. "Pan-STARRS" → "ps1")
LABEL_TO_KEY = {v["label"]: k for k, v in SURVEYS.items()}


# ---------------------------------------------------------------------------
# Coordinate parser
# ---------------------------------------------------------------------------
def parse_coords(ra_input, dec_input):
    """
    Convert RA/Dec from any common format to decimal degrees.

    Accepted RA formats
    -------------------
      Decimal degrees  :  "195.3"   or  195.3
      Colon-separated  :  "13:02:48.70"
      Space-separated  :  "13 02 48.70"

    Accepted Dec formats
    --------------------
      Decimal degrees  :  "-23.45"     or  -23.45
      Colon-separated  :  "-23:27:09.0"
      Space-separated  :  "-23 27 09.0"

    Parameters
    ----------
    ra_input : str or float
    dec_input : str or float

    Returns
    -------
    ra_deg  : float   – right ascension in decimal degrees
    dec_deg : float   – declination in decimal degrees
    """
    # Normalise space-separated sexagesimal → colon-separated so that
    # SkyCoord can parse it reliably.
    ra_str  = str(ra_input).strip().replace(" ", ":")
    dec_str = str(dec_input).strip().replace(" ", ":")

    if ":" in ra_str:
        # Sexagesimal input (hh:mm:ss / dd:mm:ss)
        coord = SkyCoord(ra=ra_str, dec=dec_str,
                         frame="icrs", unit=(u.hourangle, u.deg))
    else:
        # Decimal-degree input
        coord = SkyCoord(ra=float(ra_str), dec=float(dec_str),
                         frame="icrs", unit=(u.deg, u.deg))

    return coord.ra.deg, coord.dec.deg


# ---------------------------------------------------------------------------
# Pan-STARRS helpers
# ---------------------------------------------------------------------------
def _ps1_filenames(ra, dec, filt):
    """Query the PS1 filename service; returns an astropy Table."""
    url = (f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
           f"?ra={ra}&dec={dec}&filters={filt}")
    return Table.read(url, format="ascii")


def _ps1_cutout_url(ra, dec, size_pix, filt):
    """
    Build the URL for a Pan-STARRS FITS cutout.

    Parameters
    ----------
    ra, dec   : float   – decimal degrees
    size_pix  : int     – image side in pixels (0.25 arcsec/pixel)
    filt      : str     – filter character, e.g. 'r'

    Returns
    -------
    str or None  – cutout URL, or None if coverage is absent
    """
    table = _ps1_filenames(ra, dec, filt)
    if len(table) == 0:
        return None
    base = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size={size_pix}&format=fits"
            f"&output_size={size_pix}")
    return base + "&red=" + table["filename"][0]


# ---------------------------------------------------------------------------
# Per-catalogue FITS fetchers
# ---------------------------------------------------------------------------
def _fetch_hips2fits(hips_id, ra, dec, size_arcmin, width=512):
    """
    Fetch a FITS cutout from the CDS HiPS2FITS service (alasky.u-strasbg.fr).

    This service is fast and reliable because it re-projects HiPS sky tiles
    on-the-fly and returns a TAN-projected FITS with a full WCS header.
    It is used for Legacy Survey and GAIA which are slow on their native servers.

    Parameters
    ----------
    hips_id     : str   – HiPS identifier (e.g. 'CDS/P/DESI-Legacy-Surveys/DR10/r')
    ra, dec     : float – decimal degrees
    size_arcmin : float – field size in arcminutes
    width       : int   – output image side in pixels (square)

    Returns
    -------
    astropy.io.fits.HDUList
    """
    fov_deg = size_arcmin / 60.0
    url = (
        f"https://alasky.u-strasbg.fr/hips-image-services/hips2fits?"
        f"hips={urllib.parse.quote(hips_id, safe='')}"
        f"&ra={ra:.6f}&dec={dec:.6f}"
        f"&fov={fov_deg:.6f}"
        f"&width={width}&height={width}"
        f"&format=fits&projection=TAN"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return fits.open(BytesIO(resp.content))


def _fetch_legacy(ra, dec, size_arcmin, filt):
    """
    Download a Legacy Survey DR10 FITS cutout directly from legacysurvey.org.

    Uses the native cutout API which returns a FITS file with a proper WCS
    header. The pixel scale is 0.262 arcsec/px (fixed for Legacy Survey DR10).

    Parameters
    ----------
    ra, dec     : float – decimal degrees
    size_arcmin : float – field size in arcminutes
    filt        : str   – band letter: 'g', 'r', 'i', 'z'
    """
    # Convert field size from arcminutes to pixels at 0.262 arcsec/px
    pix_scale  = 0.262              # arcsec per pixel
    size_arcsec = size_arcmin * 60.0
    npix        = int(round(size_arcsec / pix_scale))
    npix        = max(64, min(npix, 3000))   # clamp to valid range

    url = (
        "https://www.legacysurvey.org/viewer/fits-cutout"
        f"?ra={ra}&dec={dec}&size={npix}&layer=ls-dr10&pixscale={pix_scale}&bands={filt}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    hdul = fits.open(io.BytesIO(resp.content))
    return hdul


def _fetch_ukidss(ra, dec, size_arcmin, filt):
    """
    Download a UKIDSS FITS image via the astroquery.ukidss interface.

    UKIDSS (UKIRT Infrared Deep Sky Survey) covers the northern sky in
    Y, J, H and K bands.  It is not available on SkyView, so we query
    the WFCAM Science Archive (WSA) directly through astroquery.

    Note: coverage is patchy and only for Dec > -5 deg.

    The WSA returns Multi-Extension FITS (MEF) files where:
      - Extension 0 : primary HDU, often empty (data = None)
      - Extension 1+: image planes, possibly 3-D data cubes

    We iterate over all extensions and pick the first one that contains
    a squeezable 2-D image, then return a minimal single-extension
    HDUList so the rest of the pipeline can treat it uniformly.

    Parameters
    ----------
    ra, dec     : float – decimal degrees
    size_arcmin : float – field size in arcminutes
    filt        : str   – waveband letter: 'J', 'H', or 'K'
    """
    from astroquery.ukidss import Ukidss
    coord  = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    # 'LAS' = Large Area Survey, the widest-coverage UKIDSS programme.
    images = Ukidss.get_images(
        coord,
        waveband=filt,
        image_width=size_arcmin * u.arcmin,
        programme_id="LAS",
    )
    if not images:
        raise RuntimeError(
            "No UKIDSS coverage at this position. "
            "UKIDSS covers the northern sky only (Dec > -5 deg)."
        )

    raw_hdul = images[0]  # astroquery returns a list of HDULists

    # Search through all extensions for one that contains a usable 2-D image.
    # UKIDSS MEF files often store the science image in extension 1, and the
    # primary HDU (extension 0) has no data.
    for hdu in raw_hdul:
        if hdu.data is None:
            continue
        squeezed = np.squeeze(hdu.data)
        if squeezed.ndim == 2:
            # Wrap in a new HDUList so the rest of the pipeline always
            # accesses the image via hdul[0].data / hdul[0].header.
            return fits.HDUList([fits.PrimaryHDU(data=squeezed, header=hdu.header)])

    raise RuntimeError(
        "UKIDSS returned a FITS file with no usable 2-D image plane. "
        "Try a different filter or sky position."
    )


def _fetch_ps1(ra, dec, size_arcmin, filt):
    """Download a Pan-STARRS FITS cutout."""
    size_pix = int(size_arcmin * SURVEYS["ps1"]["size_scaling"])
    url = _ps1_cutout_url(ra, dec, size_pix, filt)
    if url is None:
        raise RuntimeError("No Pan-STARRS coverage at this position.")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return fits.open(BytesIO(resp.content))


def _fetch_skymapper(ra, dec, size_arcmin, filt):
    """Download a SkyMapper (DR4) FITS cutout (southern sky only)."""
    size_deg = size_arcmin * SURVEYS["skymapper"]["size_scaling"]
    # SkyMapper accepts a CSV query; the actual FITS URL is inside the response.
    query_url = (f"https://api.skymapper.nci.org.au/public/siap/dr4/query?"
                 f"POS={ra:.4f},{dec:.4f}&SIZE={size_deg:.3f}"
                 f"&BAND={filt}&FORMAT=image/fits"
                 f"&INTERSECT=covers&RESPONSEFORMAT=CSV")
    resp = requests.get(query_url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise RuntimeError("No SkyMapper coverage at this position.")
    fits_resp = requests.get(df.iloc[0]["get_fits"], timeout=30)
    fits_resp.raise_for_status()
    return fits.open(BytesIO(fits_resp.content))


def _fetch_skyview(survey_name, ra, dec, size_arcmin):
    """
    Fetch a FITS image from NASA SkyView via the astroquery interface.

    This replaces the old raw CGI approach (runquery.pl) which was
    extremely slow and unreliable.  astroquery.skyview handles server
    communication, retries and FITS parsing internally.

    Parameters
    ----------
    survey_name : str   – survey identifier recognised by SkyView
                          (e.g. "2MASS-J", "2MASS-H", "2MASS-K",
                           "UKIDSS-J", "UKIDSS-H", "UKIDSS-K",
                           "Gaia-EDR3")
    ra, dec     : float – decimal degrees
    size_arcmin : float – field size in arcminutes

    Returns
    -------
    astropy.io.fits.HDUList
    """
    from astroquery.skyview import SkyView as _SkyView

    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    images = _SkyView.get_images(
        position   = coord,
        survey     = [survey_name],
        radius     = (size_arcmin / 2.0) * u.arcmin,
        pixels     = 512,
        show_progress = False,
    )
    if not images:
        raise RuntimeError(
            f"SkyView returned no data for survey '{survey_name}'. "
            "This sky position may not be covered by that survey."
        )
    return images[0]


# ---------------------------------------------------------------------------
# Central dispatcher: selects the right fetcher for a given catalogue key
# ---------------------------------------------------------------------------
def fetch_fits(catalog_key, ra, dec, size_arcmin, filt):
    """
    Retrieve a FITS image from the requested catalogue.

    Parameters
    ----------
    catalog_key  : str   – internal key (e.g. 'legacy', 'ps1', '2mass', 'gaia')
    ra, dec      : float – decimal degrees
    size_arcmin  : float – field size in arcminutes
    filt         : str   – filter name

    Returns
    -------
    astropy.io.fits.HDUList

    Raises
    ------
    ValueError   – unknown catalog_key
    RuntimeError – survey returned no usable data
    """
    if catalog_key == "legacy":
        return _fetch_legacy(ra, dec, size_arcmin, filt)

    elif catalog_key == "ps1":
        return _fetch_ps1(ra, dec, size_arcmin, filt)

    elif catalog_key == "skymapper":
        return _fetch_skymapper(ra, dec, size_arcmin, filt)

    elif catalog_key == "2mass":
        # SkyView survey names: "2MASS-J", "2MASS-H", "2MASS-Ks"
        # Note: K-band is called "Ks" (K-short) in the SkyView identifier.
        filt_sv = "Ks" if filt.upper() == "K" else filt.upper()
        return _fetch_skyview(f"2MASS-{filt_sv}", ra, dec, size_arcmin)

    elif catalog_key == "ukidss":
        # UKIDSS is not available on SkyView; query the WSA directly.
        return _fetch_ukidss(ra, dec, size_arcmin, filt)

    else:
        raise ValueError(f"Unknown catalogue key: '{catalog_key}'")


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------
def create_finding_chart(ra_input, dec_input, name,
                         catalog_key, filt,
                         size=3.5, errorbox=5):
    """
    Generate a finding chart and save it as a JPEG to /tmp/.

    Steps
    -----
    1. Parse the input coordinates to decimal degrees.
    2. Fetch the FITS image from the selected survey.
    3. Render with matplotlib using a WCS projection.
    4. Overlay:
         – target marker (red open circle, diameter = errorbox)
         – object name label (white)
         – North arrow (red) and East arrow (red))
         – scale bar (1/4 of the field size, red, bottom-right)
         – metadata text box (bottom-left, white background)
    5. Save the result to /tmp/<name>_<catalog>_<filter>.jpg.

    Parameters
    ----------
    ra_input    : str or float   – right ascension (any standard format)
    dec_input   : str or float   – declination (any standard format)
    name        : str            – target name used in titles and filename
    catalog_key : str            – internal catalogue key (e.g. 'legacy')
    filt        : str            – filter (e.g. 'r', 'J')
    size        : float          – field size in arcminutes (default 3.5)
    errorbox    : float          – marker area in matplotlib scatter units (default 100)

    Returns
    -------
    str  – full path to the saved JPEG file
    """
    # 1. Coordinate conversion
    ra, dec = parse_coords(ra_input, dec_input)

    # 2. Image download
    fits_data = fetch_fits(catalog_key, ra, dec, size, filt)
    if fits_data is None:
        raise RuntimeError("Survey server returned no FITS data.")

    # Squeeze out any degenerate leading dimensions returned by some surveys.
    # SkyView (used for GAIA, 2MASS, UKIDSS) often returns shape (1, N, M);
    # matplotlib requires a plain 2-D (N, M) array to display an image.
    data = np.squeeze(fits_data[0].data)
    if data.ndim != 2:
        raise RuntimeError(
            f"Unexpected image shape {data.shape} from survey. "
            "The server may have returned a data cube or an empty response."
        )
    header = fits_data[0].header
    # .celestial drops any spectral/Stokes axes so the WCS matches the 2-D array.
    wcs    = WCS(header).celestial

    survey_label = SURVEYS[catalog_key]["label"]

    # 3. Figure setup with WCS-aware axis
    fig, ax = plt.subplots(
        1, 1,
        subplot_kw={"projection": wcs},
        figsize=(7, 7),
        constrained_layout=True,
        dpi=150,
    )

    # 4a. Image display – ZScale stretch (same default as DS9)
    vmin, vmax = ZScaleInterval().get_limits(data)
    ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower", cmap="gray_r")

    # 4b. Target marker – red crosshair (open circle + four tick bars) at the
    # exact RA/Dec position.  The circle radius equals `errorbox` arcsec;
    # tick bars extend a further 50 % of the radius outward at N, S, E, W.
    #
    # We convert the sky position to pixel coordinates first so that patches
    # and lines can be drawn in data (pixel) space, which is simpler and more
    # predictable than the WCS transform for geometric shapes.
    px, py = wcs.world_to_pixel(SkyCoord(ra=float(ra)*u.deg, dec=float(dec)*u.deg))
    px, py = float(px), float(py)

    # Convert errorbox from arcsec to pixels using the real WCS pixel scale,
    # so that the circle radius matches the actual error region on the sky.
    # proj_plane_pixel_scales returns degrees/pixel; multiply by 3600 for arcsec/pixel.
    _pix_scales_early = proj_plane_pixel_scales(wcs) * 3600.0   # arcsec/pixel
    _pix_scale_early  = float(np.mean(_pix_scales_early))       # mean of both axes
    radius_pix = max(1, int(round(errorbox / _pix_scale_early)))  # errorbox arcsec → pixels
    tick_pix   = max(1, int(radius_pix * 0.5))  # tick bars = 50 % of radius, min 1px

    # Open circle centred on the target.
    ax.add_patch(mpatches.Circle(
        (px, py), radius=radius_pix,
        edgecolor="deepskyblue", facecolor="none", lw=3,
        transform=ax.transData,
    ))

    # Four tick bars extending outward from the edge of the circle.
    for dx, dy in [
        (0,  1),   # North
        (0, -1),   # South
        (1,  0),   # East
        (-1, 0),   # West
    ]:
        ax.plot(
            [px + dx * radius_pix, px + dx * (radius_pix + tick_pix)],
            [py + dy * radius_pix, py + dy * (radius_pix + tick_pix)],
            color="red", lw=2, transform=ax.transData,
        )

    # 4c. Target name label – centered horizontally above the crosshair.
    # We work in pixel (data) coordinates so the offset is proportional to
    # the marker radius, guaranteeing the label never overlaps the circle
    # regardless of the errorbox size or the survey pixel scale.
    # Gap = radius + 30 % extra so the label clears the top tick bar.
    label_gap = radius_pix + tick_pix + max(4, int(radius_pix * 0.6))
    ax.text(
        px, py + label_gap, name,
        transform=ax.transData,
        fontsize="large",
        color="white",
        fontweight="bold",
        ha="center", va="bottom",
    )

    # 4d. Scale bar – drawn manually in axes-fraction coordinates so it is
    # always positioned correctly regardless of image size or resolution.
    # The bar represents 1/4 of the total field; the label sits inside a
    # semi-transparent cyan panel consistent with the metadata box style.
    ny, nx           = data.shape
    # Derive actual pixel scale from the WCS (more accurate than the nominal).
    pix_scales       = proj_plane_pixel_scales(wcs) * 3600.0   # arcsec/pixel
    pix_scale_arcsec = float(np.max(pix_scales))
    scalebar_arcsec  = (size * 60.0) / 4.0                     # 1/4 of the field
    scalebar_pix     = scalebar_arcsec / pix_scale_arcsec      # bar length in pixels
    # Convert to an axes-fraction width; clamp to a sensible 6–38 % range.
    scalebar_ax_frac = float(np.clip(scalebar_pix / nx, 0.06, 0.38))
    x_right = 0.96
    x_left  = x_right - scalebar_ax_frac
    y_bar   = 0.055
    # Horizontal bar with end ticks drawn in red.
    ax.annotate(
        "", xy=(x_right, y_bar), xytext=(x_left, y_bar),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="|-|", color="red", lw=2),
        zorder=5,
    )
    # Label centred above the bar inside a semi-transparent grey panel.
    ax.text(
        (x_left + x_right) / 2, y_bar + 0.04,
        f'{scalebar_arcsec:.0f}"',
        ha="center", va="bottom", fontsize=9,
        color="red", fontweight="bold",
        transform=ax.transAxes, zorder=6,
        bbox=dict(facecolor="lightgrey", edgecolor="none",
                  boxstyle="round,pad=0.2", alpha=0.6),
    )

    # 4e. Compass arrows – drawn in axes-fraction coordinates so they are
    # always fully visible regardless of image size, zoom level, or pixel scale.
    # North arrow is red; East arrow is cyan for easy visual distinction.
    # Both labels have a dark semi-transparent background for legibility.
    x0_ax  = 0.87   # compass origin: 87 % from the left edge of the axes
    y0_ax  = 0.80   # compass origin: 80 % from the bottom edge
    len_ax = 0.08   # arrow length:   8 % of the axes side

    # North arrow – pointing upward (red)
    ax.annotate(
        "",
        xy=(x0_ax, y0_ax + len_ax), xytext=(x0_ax, y0_ax),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(facecolor="red", edgecolor="none",
                        width=1.5, headwidth=9, headlength=11),
        zorder=10,
    )
    ax.text(
        x0_ax, y0_ax + len_ax + 0.025, "N",
        ha="center", va="bottom",
        fontsize=11, color="red", fontweight="bold",
        transform=ax.transAxes, zorder=10,
    )

    # East arrow – pointing leftward (cyan)
    ax.annotate(
        "",
        xy=(x0_ax - len_ax, y0_ax), xytext=(x0_ax, y0_ax),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(facecolor="red", edgecolor="none",
                        width=1.5, headwidth=9, headlength=11),
        zorder=10,
    )
    ax.text(
        x0_ax - len_ax - 0.025, y0_ax, "E",
        ha="right", va="center",
        fontsize=11, color="red", fontweight="bold",
        transform=ax.transAxes, zorder=10,
    )

    # 4f. Metadata text box – bottom-left corner, semi-transparent white panel.
    # Shows object name, survey, field size, RA, Dec and pixel scale.
    ax.text(
        0.03, 0.10,
        (f"{name}  —  {survey_label}   {size:.1f}' \u00d7 {size:.1f}'\n"
         f"RA  = {ra_input}  ({ra:.4f}\u00b0)\n"
         f"Dec = {dec_input}  ({dec:.4f}\u00b0)\n"
         f"Pix scale = {pix_scale_arcsec:.2f}\"/pix"),
        transform=ax.transAxes,
        fontsize="small", va="center", ha="left",
        color="black", zorder=4,
        bbox=dict(facecolor="white", edgecolor="black",
                  boxstyle="round,pad=0.2", alpha=0.85),
    )

    # 4g. Axis labels, title and coordinate grid
    filt_display = filt.upper() if filt.lower() in ("j", "h", "k") else filt
    ax.set_xlabel("RA",  fontsize="large")
    ax.set_ylabel("Dec", fontsize="large")
    ax.set_title(
        f"${filt_display}$-band finding chart  —  {name}",
        fontsize="x-large", pad=20,
    )
    ax.coords.grid(True, color="black", linewidth=0.5)

    # 5. Save to /tmp
    safe_name = (name.replace(".", "").replace("-", "")
                     .replace(" ", "_").replace(":", ""))
    fc_filename = f"{safe_name}_{catalog_key}_{filt}.jpg"
    out_path    = os.path.join("/tmp", fc_filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------
class FindingChartApp(tk.Tk):
    """
    Main application window.

    Layout
    ------
    Left panel  : input form (name, RA, Dec, catalogue dropdown, filter
                  dropdown, field size, error-box) plus Generate / Save buttons
                  and a status line.
    Right panel : canvas label that displays a scaled preview of the chart.

    The filter dropdown is dynamic: its options update automatically whenever
    the user selects a different catalogue, showing only the filters that are
    actually available for that survey.
    """

    # Default values pre-filled in the entry widgets at startup.
    DEFAULTS = {
        "name":     "MyTarget",
        "ra":       "03:33:48.140",
        "dec":      "-19:29:44.52",
        "catalog":  "Legacy Survey",
        "size":     "3.5",
        "errorbox": "5",
    }

    def __init__(self):
        super().__init__()
        self.title("Finding Chart Generator")
        self.resizable(True, True)
        # Set an initial window size and a minimum so the preview panel
        # never collapses when its content changes.
        self.geometry("1200x750")
        self.minsize(900, 600)

        # Path of the most recently generated JPEG (used by the Save button).
        self._last_chart_path = None
        # Strong reference to PhotoImage so it is not garbage-collected.
        self._photo = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI builder
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Create and lay out all widgets inside the main window."""

        # ---- Two top-level panels: left (form) and right (preview) --------
        left  = tk.Frame(self, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = tk.Frame(self, padx=10, pady=10, bg="#1a1a1a")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Prevent the frame from shrinking to the size of its child widgets
        # when a small image is loaded into the preview canvas.
        right.pack_propagate(False)

        # ---- Section title ------------------------------------------------
        tk.Label(left, text="Finding Chart Parameters",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 8))

        # ---- Helper: one labelled Entry row --------------------------------
        # Returns (StringVar, Entry) so callers can bind keyboard events.
        def add_entry(label, default):
            tk.Label(left, text=label, anchor="w").pack(fill="x")
            var    = tk.StringVar(value=default)
            widget = tk.Entry(left, textvariable=var, width=30)
            widget.pack(fill="x", pady=(0, 6))
            # Pressing Enter in any field triggers chart generation.
            widget.bind("<Return>", lambda _e: self._on_generate())
            return var

        self.var_name     = add_entry("Object name:", self.DEFAULTS["name"])
        self.var_ra       = add_entry("RA  (hh:mm:ss  or  degrees):",
                                      self.DEFAULTS["ra"])
        self.var_dec      = add_entry("Dec  (dd:mm:ss  or  degrees):",
                                      self.DEFAULTS["dec"])

        # ---- Catalogue dropdown --------------------------------------------
        tk.Label(left, text="Catalogue:", anchor="w").pack(fill="x")
        catalog_labels   = [v["label"] for v in SURVEYS.values()]
        self.var_catalog = tk.StringVar(value=self.DEFAULTS["catalog"])
        self.cb_catalog  = ttk.Combobox(
            left, textvariable=self.var_catalog,
            values=catalog_labels, state="readonly", width=28,
        )
        self.cb_catalog.pack(fill="x", pady=(0, 6))
        # Bind catalogue selection change to the filter updater.
        self.cb_catalog.bind("<<ComboboxSelected>>", self._on_catalog_change)
        # Pressing Enter on the catalogue dropdown also triggers generation.
        self.cb_catalog.bind("<Return>", lambda _e: self._on_generate())

        # ---- Filter dropdown (contents change with catalogue) --------------
        tk.Label(left, text="Filter:", anchor="w").pack(fill="x")
        self.var_filter = tk.StringVar()
        self.cb_filter  = ttk.Combobox(
            left, textvariable=self.var_filter,
            state="readonly", width=28,
        )
        self.cb_filter.pack(fill="x", pady=(0, 6))
        # Pressing Enter on the filter dropdown also triggers generation.
        self.cb_filter.bind("<Return>", lambda _e: self._on_generate())
        # Populate immediately for the default catalogue.
        self._update_filter_dropdown(self.DEFAULTS["catalog"])

        # ---- Numeric parameters -------------------------------------------
        self.var_size     = add_entry("Field size (arcmin, default 3.5):",
                                       self.DEFAULTS["size"])
        self.var_errorbox = add_entry("Error-box radius (arcsec, default 5):",
                                       self.DEFAULTS["errorbox"])

        # ---- Action buttons -----------------------------------------------
        # Keep a reference to the Generate button so we can disable it
        # while the background thread is running, preventing double clicks.
        self.btn_generate = tk.Button(
            left, text="Generate Finding Chart",
            command=self._on_generate,
            bg="#4a90d9", fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.RAISED, padx=6, pady=4,
        )
        self.btn_generate.pack(fill="x", pady=(12, 4))

        self.btn_save = tk.Button(
            left, text="Save to Downloads",
            command=self._on_save,
            state=tk.DISABLED,
            padx=6, pady=4,
        )
        self.btn_save.pack(fill="x", pady=(0, 6))

        # ---- Status line --------------------------------------------------
        self.var_status = tk.StringVar(value="Ready.")
        tk.Label(
            left, textvariable=self.var_status,
            fg="gray", wraplength=230, justify="left",
        ).pack(anchor="w", pady=(6, 0))

        # ---- Preview canvas (right panel) ---------------------------------
        tk.Label(
            right, text="Preview",
            font=("Helvetica", 12, "bold"),
            bg="#1a1a1a", fg="white",
        ).pack(anchor="w")

        self.canvas = tk.Label(
            right,
            text="(chart will appear here)",
            bg="#1a1a1a", fg="#555555",
            width=60, height=30,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_catalog_change(self, _event=None):
        """Called whenever the user picks a different catalogue."""
        self._update_filter_dropdown(self.var_catalog.get())

    def _update_filter_dropdown(self, catalog_label):
        """
        Rebuild the filter Combobox values for the selected catalogue.

        Parameters
        ----------
        catalog_label : str  – display label of the catalogue (e.g. 'Pan-STARRS')
        """
        key = LABEL_TO_KEY.get(catalog_label)
        if key is None:
            return
        filters = SURVEYS[key]["filters"]
        self.cb_filter["values"] = filters
        self.var_filter.set(filters[0])    # default to the first filter

    def _on_generate(self):
        """
        Read the input fields, validate them, call create_finding_chart(),
        and display the result in the preview panel.
        """
        # --- Read inputs ---------------------------------------------------
        name        = self.var_name.get().strip()
        ra_raw      = self.var_ra.get().strip()
        dec_raw     = self.var_dec.get().strip()
        catalog_lbl = self.var_catalog.get()
        filt        = self.var_filter.get()

        # --- Input validation ----------------------------------------------
        if not name:
            messagebox.showerror("Input error", "Please enter an object name.")
            return
        if not ra_raw or not dec_raw:
            messagebox.showerror("Input error", "Please enter RA and Dec.")
            return

        try:
            size = float(self.var_size.get())
            if size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input error",
                                  "Field size must be a positive number (arcmin).")
            return

        try:
            errorbox = float(self.var_errorbox.get())
            if errorbox < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input error",
                                  "Error-box radius must be a non-negative number.")
            return

        catalog_key = LABEL_TO_KEY.get(catalog_lbl)
        if catalog_key is None:
            messagebox.showerror("Input error", "Unknown catalogue selected.")
            return

        # --- Disable buttons and show feedback while the thread runs -------
        self.var_status.set("Downloading image and generating chart...")
        self.btn_save.config(state=tk.DISABLED)
        self.btn_generate.config(state=tk.DISABLED)
        self.update_idletasks()

        # --- Launch generation in a background thread ---------------------
        # Running create_finding_chart() in a thread keeps the tkinter event
        # loop alive (so the window stays responsive) while the network
        # request and matplotlib rendering complete.
        thread = threading.Thread(
            target=self._run_generate,
            args=(ra_raw, dec_raw, name, catalog_key, filt, size, errorbox),
            daemon=True,
        )
        thread.start()

    def _run_generate(self, ra_raw, dec_raw, name, catalog_key, filt, size, errorbox):
        """
        Worker method executed in a background thread.
        Calls create_finding_chart() and schedules the result callback
        on the main thread via self.after().
        """
        try:
            chart_path = create_finding_chart(
                ra_input    = ra_raw,
                dec_input   = dec_raw,
                name        = name,
                catalog_key = catalog_key,
                filt        = filt,
                size        = size,
                errorbox    = errorbox,
            )
            # Schedule success callback on the main (tkinter) thread.
            self.after(0, self._on_generate_success, chart_path)
        except Exception as exc:
            # Schedule error callback on the main thread.
            self.after(0, self._on_generate_error, str(exc))

    def _on_generate_success(self, chart_path):
        """Called from the main thread after successful chart generation."""
        self._last_chart_path = chart_path
        self._show_preview(chart_path)
        self.var_status.set(f"Chart ready: {os.path.basename(chart_path)}")
        self.btn_save.config(state=tk.NORMAL)
        self.btn_generate.config(state=tk.NORMAL)

    def _on_generate_error(self, message):
        """Called from the main thread when chart generation raises an exception."""
        self.var_status.set(f"Error: {message}")
        messagebox.showerror("Chart generation failed", message)
        self.btn_generate.config(state=tk.NORMAL)

    def _show_preview(self, path):
        """
        Load the JPEG at `path` and display it in the preview canvas.
        The image is scaled to fill the available canvas area while
        preserving its aspect ratio.

        We call self.update() first to ensure tkinter has processed any
        pending geometry changes, so winfo_width/height return the real
        current pixel dimensions of the canvas.
        """
        # Flush pending layout so winfo_* returns current pixel dimensions.
        self.update()
        img = Image.open(path)
        # Use at least 600 × 600 px as a safe fallback.
        canvas_w = max(self.canvas.winfo_width(),  600)
        canvas_h = max(self.canvas.winfo_height(), 600)
        img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)

        # Store in an instance variable to prevent garbage collection.
        self._photo = ImageTk.PhotoImage(img)
        # width=1/height=1 prevent the Label from resizing itself to
        # match the image; the pack geometry manager handles sizing instead.
        self.canvas.config(image=self._photo, text="", width=1, height=1)

    def _on_save(self):
        """
        Copy the last generated JPEG to the user's ~/Downloads folder.
        The file is saved with the same filename as the /tmp original.
        """
        if not self._last_chart_path or not os.path.isfile(self._last_chart_path):
            messagebox.showerror("Save error", "No chart available to save.")
            return

        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads, exist_ok=True)
        dest = os.path.join(downloads, os.path.basename(self._last_chart_path))

        with open(self._last_chart_path, "rb") as src_f, \
             open(dest, "wb") as dst_f:
            dst_f.write(src_f.read())

        self.var_status.set(f"Saved to: {dest}")
        messagebox.showinfo("Saved", f"Finding chart saved to:\n{dest}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = FindingChartApp()
    app.mainloop()
