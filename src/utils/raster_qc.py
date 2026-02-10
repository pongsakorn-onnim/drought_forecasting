# src/utils/raster_qc.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import math


@dataclass(frozen=True)
class RasterMeta:
    """
    Immutable container for raster file metadata extracted from GeoTIFF headers.
    
    Stores geospatial properties, file attributes, and temporal information parsed
    from raster filenames. All fields are read-only (frozen) to ensure data integrity
    during QC pipeline execution.
    
    Attributes
    ----------
    path : str
        Absolute or relative file path as string.
        Example: "data/raw/obs_monthly_tif/o_th202301_fixed.tif"
        
    filename : str
        Base filename without directory path.
        Example: "o_th202301_fixed.tif"
        
    month_key : str
        6-digit month identifier (YYYYMM) extracted from filename, or "UNKNOWN".
        Example: "202301" (January 2023)
        
    variant : str
        Processing variant: "fixed" (corrected), "raw" (original), or "unknown".
        Determined from '_fixed' suffix in filename.
        
    driver : str
        Raster format driver (e.g., 'GTiff' for GeoTIFF).
        
    crs : str
        Coordinate Reference System as string representation.
        Example: "EPSG:32647" (UTM zone 47N), "EPSG:4326" (WGS84), or "NONE".
        
    width : int
        Number of columns (x-dimension) in pixels.
        Example: 1000 (1000 pixels wide)
        
    height : int
        Number of rows (y-dimension) in pixels.
        Example: 800 (800 pixels tall)
        
    count : int
        Number of bands in the raster.
        Example: 1 (single-band), 3 (RGB), 4 (RGBA)
        
    dtype : str
        Data type of pixel values.
        Common values: 'uint8', 'uint16', 'int16', 'float32', 'float64'
        
    nodata : Optional[float]
        Value representing NoData/missing pixels.
        Examples: -9999.0, 0.0, None (if undefined)
        
    transform : Tuple[float, float, float, float, float, float]
        6-parameter affine transformation matrix (a, b, c, d, e, f) for 
        pixel-to-geographic coordinate conversion.
        
        Format: (a, b, c, d, e, f) corresponds to:
        [ a  b  c ]   [ pixel width   rotation      x-coordinate of upper-left ]
        [ d  e  f ] = [ rotation      pixel height  y-coordinate of upper-left ]
        [ 0  0  1 ]   [ 0             0             1                          ]
        
        Example: (30.0, 0.0, 100000.0, 0.0, -30.0, 200000.0)
        - a=30.0: pixel width (x-resolution) in CRS units
        - e=-30.0: pixel height (negative because y-axis decreases downward)
        - c=100000.0: x-coordinate of upper-left corner
        - f=200000.0: y-coordinate of upper-left corner
        
    bounds : Tuple[float, float, float, float]
        Geographic bounding box in CRS units: (left, bottom, right, top).
        Example: (100000.0, 170000.0, 130000.0, 200000.0)
        - left: minimum x-coordinate (west)
        - bottom: minimum y-coordinate (south)
        - right: maximum x-coordinate (east)
        - top: maximum y-coordinate (north)
        
    res : Tuple[float, float]
        Pixel resolution (x_resolution, y_resolution) in CRS units.
        Example: (30.0, 30.0) for 30m x 30m pixels
        Note: y_resolution is typically negative for north-up rasters.
        
    Examples
    --------
    >>> meta = RasterMeta(
    ...     path="data/raw/o_th202301_fixed.tif",
    ...     filename="o_th202301_fixed.tif",
    ...     month_key="202301",
    ...     variant="fixed",
    ...     driver="GTiff",
    ...     crs="EPSG:32647",
    ...     width=1000,
    ...     height=800,
    ...     count=1,
    ...     dtype="float32",
    ...     nodata=-9999.0,
    ...     transform=(30.0, 0.0, 100000.0, 0.0, -30.0, 200000.0),
    ...     bounds=(100000.0, 170000.0, 130000.0, 200000.0),
    ...     res=(30.0, 30.0)
    ... )
    >>> meta.month_key
    '202301'
    >>> meta.res
    (30.0, 30.0)
    >>> f"Image size: {meta.width}x{meta.height} pixels"
    'Image size: 1000x800 pixels'
    
    Notes
    -----
    1. This class is frozen (immutable) - instances cannot be modified after creation.
    2. Transform and bounds are derived from each other; both are stored for convenience.
    3. For time-series analysis, ensure consistent `month_key` format across all files.
    4. The `variant` field helps identify corrected ('fixed') vs original ('raw') data.
    5. `nodata` value is critical for statistical calculations and masking.
    
    See Also
    --------
    read_raster_meta : Function to create RasterMeta from raster file
    scan_raster_metas : Bulk extraction of RasterMeta objects
    """
    path: str
    filename: str
    month_key: str  # YYYYMM (derived from filename pattern)
    variant: str  # "fixed" or "raw" or "unknown"

    driver: str
    crs: str
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]

    transform: Tuple[float, float, float, float, float, float]  # (a, b, c, d, e, f)
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    res: Tuple[float, float]  # (xres, yres)


def _infer_month_key_and_variant(path: Path) -> Tuple[str, str]:
    """
    Parse month key and variant from raster filename.
    
    Internal helper function that extracts temporal metadata from
    filenames following project convention.
    
    Parameters
    ----------
    path : Path
        Path object for raster file.
    
    Returns
    -------
    Tuple[str, str]
        (month_key, variant)
        - month_key: "YYYYMM" or "UNKNOWN"
        - variant: "fixed", "raw", or "unknown"
    
    Notes
    -----
    Filename patterns:
    - o_thYYYYMM.tif → ("YYYYMM", "raw")
    - o_thYYYYMM_fixed.tif → ("YYYYMM", "fixed")
    - Other patterns → ("UNKNOWN", "raw")
    
    This logic is critical for monthly time-series organization.
    
    Examples
    --------
    >>> _infer_month_key_and_variant(Path("o_th202301.tif"))
    ('202301', 'raw')
    >>> 
    >>> _infer_month_key_and_variant(Path("o_th202302_fixed.tif"))
    ('202302', 'fixed')
    >>> 
    >>> _infer_month_key_and_variant(Path("other_file.tif"))
    ('UNKNOWN', 'raw')
    """
    stem = path.stem  # without suffix .tif
    variant = "unknown"
    base = stem

    if stem.endswith("_fixed"):
        variant = "fixed"
        base = stem[:-6]  # remove "_fixed"
    else:
        variant = "raw"

    # month_key: last 6 digits in base (e.g., o_th202512 -> 202512)
    # keep strict: require exactly 6 digits at end
    month_key = "UNKNOWN"
    if len(base) >= 6 and base[-6:].isdigit():
        month_key = base[-6:]

    return month_key, variant


def list_rasters(folder: Union[str, Path], pattern: str = "o_th*.tif", recursive: bool = False) -> List[Path]:
    """
    Discover raster files in directory matching specified pattern.
    
    First step in QC pipeline - finds all candidate files for processing.
    
    Parameters
    ----------
    folder : Union[str, Path]
        Directory to search.
    
    pattern : str, optional
        Glob pattern for filename matching.
        Default: "o_th*.tif" (monthly observation files)
        
        Examples:
        - "o_th2023*.tif": Only 2023 files
        - "*.tif": All TIFF files
        - "*_fixed.tif": Only fixed variant
    
    recursive : bool, optional
        If True, search subdirectories recursively.
        Default: False.
    
    Returns
    -------
    List[Path]
        Sorted list of Path objects for matching files.
    
    Raises
    ------
    FileNotFoundError
        If specified folder doesn't exist.
    
    Examples
    --------
    >>> # Find all monthly files
    >>> files = list_rasters("data/raw/obs_monthly_tif")
    >>> print(f"Found {len(files)} files")
    >>> 
    >>> # Recursive search
    >>> files = list_rasters("data/raw", recursive=True)
    >>> 
    >>> # Specific pattern
    >>> files = list_rasters("data/raw", pattern="o_th2023*.tif")
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if recursive:
        paths = sorted(folder.rglob(pattern))
    else:
        paths = sorted(folder.glob(pattern))

    return [p for p in paths if p.is_file()]


def choose_preferred_by_month(paths: Iterable[Path]) -> Tuple[Dict[str, Path], pd.DataFrame]:
    """
    Select the best file for each month based on deterministic rules.
    
    Groups files by month_key and applies selection rules:
    1. If both raw and fixed exist → choose fixed
    2. If multiple variants exist → mark as anomalies
    3. If month_key is UNKNOWN → mark as anomalies
    
    Parameters
    ----------
    paths : Iterable[Path]
        List of raster file paths
        
    Returns
    -------
    Tuple[Dict[str, Path], pd.DataFrame]
        chosen: Dictionary mapping month_key to selected Path
        df_anom: DataFrame of selection anomalies/issues
        
    Notes
    -----
    Selection priorities:
    1. fixed variant (if exists)
    2. raw variant (if no fixed)
    3. If multiple files in same variant → select last after sorting
    
    Anomaly types:
    - unknown_month_key: Cannot extract YYYYMM from filename
    - unknown_variant: Variant not "raw" or "fixed"
    - duplicate_fixed: Multiple fixed files for same month
    - duplicate_raw: Multiple raw files for same month
    - no_valid_file: No valid files for a month_key
    
    Examples
    --------
    >>> paths = [Path("o_th202301.tif"), Path("o_th202301_fixed.tif")]
    >>> chosen, anomalies = choose_preferred_by_month(paths)
    >>> chosen
    {"202301": Path("o_th202301_fixed.tif")}
    """
    buckets: Dict[str, List[Path]] = {}
    info: Dict[Path, Tuple[str, str]] = {}

    for p in paths:
        month_key, variant = _infer_month_key_and_variant(p)
        info[p] = (month_key, variant)
        buckets.setdefault(month_key, []).append(p)

    chosen: Dict[str, Path] = {}
    anomalies = []

    for month_key, ps in sorted(buckets.items(), key=lambda x: x[0]):
        # If unknown month key -> anomaly
        if month_key == "UNKNOWN":
            for p in ps:
                anomalies.append(
                    {
                        "month_key": month_key,
                        "issue": "unknown_month_key",
                        "path": str(p),
                    }
                )
            continue

        fixed = [p for p in ps if info[p][1] == "fixed"]
        raw = [p for p in ps if info[p][1] == "raw"]
        other = [p for p in ps if info[p][1] not in ("fixed", "raw")]

        # duplicates / unexpected
        if len(other) > 0:
            for p in other:
                anomalies.append(
                    {"month_key": month_key, "issue": "unknown_variant", "path": str(p)}
                )

        if len(fixed) > 1:
            for p in fixed:
                anomalies.append(
                    {"month_key": month_key, "issue": "duplicate_fixed", "path": str(p)}
                )
        if len(raw) > 1:
            for p in raw:
                anomalies.append(
                    {"month_key": month_key, "issue": "duplicate_raw", "path": str(p)}
                )

        # selection
        if len(fixed) >= 1:
            chosen[month_key] = sorted(fixed)[-1]  # deterministic if multiple
        elif len(raw) >= 1:
            chosen[month_key] = sorted(raw)[-1]
        else:
            # no valid file for month_key
            anomalies.append(
                {"month_key": month_key, "issue": "no_valid_file", "path": ""}
            )

    df_anom = pd.DataFrame(anomalies)
    return chosen, df_anom


def read_raster_meta(path: Union[str, Path]) -> RasterMeta:
    """
    Extract metadata from single raster file header.
    
    Fast, header-only read that extracts geospatial properties and
    file attributes without loading pixel data.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to raster file.
    
    Returns
    -------
    RasterMeta
        Complete metadata as frozen dataclass.
    
    Notes
    -----
    Extracted information:
    - Geospatial: CRS, transform, bounds, resolution
    - Structural: Dimensions, data type, band count
    - File: Driver, nodata value
    - Temporal: Month key and variant from filename
    
    Examples
    --------
    >>> meta = read_raster_meta("data/raw/o_th202301_fixed.tif")
    >>> 
    >>> # Access properties
    >>> print(f"Size: {meta.width}x{meta.height}")
    >>> print(f"CRS: {meta.crs}")
    >>> print(f"Resolution: {meta.res}")
    >>> 
    >>> # Check specific conditions
    >>> if meta.nodata is None:
    ...     print("WARNING: No nodata value defined")
    """
    path = Path(path)
    
    month_key, variant = _infer_month_key_and_variant(path)

    with rasterio.open(path) as src:
        crs = src.crs.to_string() if src.crs else "NONE"
        transform = src.transform  # Affine
        bounds = src.bounds
        res = src.res

        meta = RasterMeta(
            path=str(path),
            filename=path.name,
            month_key=month_key,
            variant=variant,
            driver=src.driver or "UNKNOWN",
            crs=crs,
            width=int(src.width),
            height=int(src.height),
            count=int(src.count),
            dtype=str(src.dtypes[0]) if src.dtypes else "UNKNOWN",
            nodata=None if src.nodata is None else float(src.nodata),
            transform=(float(transform.a), float(transform.b), float(transform.c),
                       float(transform.d), float(transform.e), float(transform.f)),
            bounds=(float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)),
            res=(float(res[0]), float(res[1])),
        )

    return meta


def scan_raster_metas(
    chosen: Dict[str, Path],
) -> pd.DataFrame:
    """
    Extract metadata for all selected raster files and compile into DataFrame.
    
    Processes the output from `choose_preferred_by_month()`, reading metadata
    for each selected file and organizing it into a tabular format suitable
    for analysis and consistency checking.
    
    Parameters
    ----------
    chosen : Dict[str, Path]
        Dictionary mapping month keys (YYYYMM) to selected file paths.
        Typically comes from `choose_preferred_by_month()` output.
        
        Example:
        {
            "202301": Path("data/raw/obs_monthly_tif/o_th202301_fixed.tif"),
            "202302": Path("data/raw/obs_monthly_tif/o_th202302.tif")
        }
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per month/file, containing all metadata fields
        from `RasterMeta`. Columns include:
        
        ['path', 'filename', 'month_key', 'variant', 'driver', 'crs',
         'width', 'height', 'count', 'dtype', 'nodata', 'transform',
         'bounds', 'res']
        
        Sorted chronologically by month_key.
    
    Notes
    -----
    - Reads only file headers (fast operation)
    - Uses `read_raster_meta()` internally for each file
    - The resulting DataFrame is the primary input for consistency checks
      and coverage analysis
    
    Examples
    --------
    >>> # From QC pipeline
    >>> files = list_rasters("data/raw/obs_monthly_tif")
    >>> chosen, _ = choose_preferred_by_month(files)
    >>> df_meta = scan_raster_metas(chosen)
    >>> 
    >>> # Inspect results
    >>> df_meta[['month_key', 'filename', 'width', 'height', 'res']].head()
    >>> 
    >>> # Check specific properties
    >>> unique_crs = df_meta['crs'].unique()
    >>> print(f"Found {len(unique_crs)} unique CRS: {unique_crs}")
    """
    metas = []
    for month_key, p in sorted(chosen.items(), key=lambda x: x[0]):
        m = read_raster_meta(p)
        metas.append(asdict(m))

    df = pd.DataFrame(metas)
    if not df.empty:
        df = df.sort_values(["month_key"]).reset_index(drop=True)
    return df

def check_value_diff(actual, expected, tol=1e-5):
    """
    Hepler function ตัดสินว่าค่าต่างกันหรือไม่ (Return True ถ้าต่างกัน)
    รองรับทั้ง Tuple, Float, Int, String    

    Args:
        actual (_type_): _description_
        expected (_type_): _description_
        tol (_type_, optional): _description_. Defaults to 1e-5.
    """
    # เช็คว่าเป็น None หรือไม่ ถ้าเป็นทั้งคู่ถือว่าไม่ต่างกัน
    if pd.isna(actual) and pd.isna(expected):
        return False
    
    # กรณี Tuple (เช่น transform, bounds)
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        if len(actual) != len(expected):
            return True
        
        # เช็ค elements แต่ละคู่ของ transform, bounds 
        return any(
            not math.isclose(a, e, rel_tol=tol)
            if (isinstance(a, (float, int)) and isinstance(e, (float, int))) # <--- เช็คคู่เลย
            else a != e
            for a, e in zip(actual, expected)
            )
    # เช็คกรณีตัวเลขเดี่ยว ๆ ทั่วไป ที่ไม่ใช่พวกตัวเลขใน tuple
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return not math.isclose(actual, expected, rel_tol=tol)
    
    # กรณีค่าทั่วไป เช่น string, etc.
    return actual != expected

def find_inconsistencies(
    df_meta, 
    fields: Optional[List[str]] = None,
    baseline_month: Optional[str] = None,) -> pd.DataFrame:
    
    """
    เปรียบเทียบค่าในคอลัมน์ต่าง ๆ (fields) ของแต่ละไฟล์ เทียบกับค่า baseline
    หากพบความแตกต่างกับ baseline จะลงบันทึกใน DataFrame แล้วส่งออกมา 
    baseline จะใช้ข้อมูลตัวแรกของ df_meta หรือใช้ตาม month_key ที่ระบุ
    """
                         
    if df_meta.empty:
        return pd.DataFrame(columns=["month_key", "field", "expected", "actual", "filename", "path"])
    
    if fields is None:
        fields = ["crs", "width", "height", "count", "dtype", "nodata", "transform", "bounds", "res"]
    
    # หา Baseline เอาไว้ใช้เทียบไฟล์อื่น ๆ 
    if baseline_month is not None and (df_meta["month_key"] == baseline_month).any():
        base_row = df_meta.loc[df_meta["month_key"] == baseline_month].iloc[0]
    else:
        base_row = df_meta.iloc[0]
    
    expected = {f: base_row[f] for f in fields}
    
    rows = []
    
    for row in df_meta.itertuples(index=False):
        for f in fields:
            act_val = getattr(row, f)
            exp_val = expected[f]
            
            if check_value_diff(act_val, exp_val):
                rows.append({
                    "month_key": row.month_key,
                    "field": f,
                    "expected": exp_val,
                    "actual": act_val,
                    "filename": getattr(row, "filename", ""),
                    "path": getattr(row, "path", ""),
                })
    
    return pd.DataFrame(rows)

def compute_month_coverage(
    month_keys: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze temporal continuity and identify missing months in dataset.
    
    Calculates the date range from available month keys and detects any
    gaps in the monthly sequence. Essential for time-series completeness
    assessment.
    
    Parameters
    ----------
    month_keys : Iterable[str]
        Collection of month identifiers in YYYYMM format.
        Typically from df_meta['month_key'].
        
        Example: ["202301", "202302", "202304"] (missing 202303)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        df_range : DataFrame with coverage summary
            ['min_month', 'max_month', 'n_months_detected',
             'n_months_expected_between_min_max', 'n_missing']
             
        df_missing : DataFrame listing missing months
            ['missing_month']
    
    Notes
    -----
    - Ignores "UNKNOWN" month keys
    - Months are inclusive in range calculation
    - Uses pandas Period for accurate month arithmetic
    
    Examples
    --------
    >>> month_keys = ["202301", "202302", "202304", "202305"]
    >>> df_range, df_missing = compute_month_coverage(month_keys)
    >>> 
    >>> print(f"Date range: {df_range['min_month'].iloc[0]} to {df_range['max_month'].iloc[0]}")
    >>> print(f"Missing {df_range['n_missing'].iloc[0]} month(s)")
    >>> 
    >>> if not df_missing.empty:
    ...     print("Missing months:", df_missing['missing_month'].tolist())
    """
    keys = sorted([k for k in set(month_keys) if isinstance(k, str) and k != "UNKNOWN"])
    if not keys:
        return (
            pd.DataFrame([{"min_month": None, "max_month": None, "n_months": 0}]),
            pd.DataFrame(columns=["missing_month"]),
        )

    def to_period(k: str) -> pd.Period:    
        """
        ฟังก์ชันแปลง string YYYYMM → pandas Period
        แปลง "202301" → pd.Period("2023-01", freq="M") 
        pandas Period ทำให้คำนวณเดือนได้ง่าย (บวก/ลบเดือน, เปรียบเทียบ)
        """                 
        return pd.Period(f"{k[:4]}-{k[4:6]}", freq="M")

    periods = [to_period(k) for k in keys]                      # periods = [Period('2023-01', 'M'), Period('2023-02', 'M'), Period('2023-04', 'M')]
    pmin, pmax = min(periods), max(periods)                 
    
    # สร้าง list เดือนแบบสมบูรณ์ระหว่าง pmin ถึง pmax
    full = pd.period_range(pmin, pmax, freq="M")                # ตัวอย่าง เช่น full = [Period('2023-01', 'M'), Period('2023-02', 'M'), ..., Period('2023-04', 'M')]
    full_keys = [f"{p.year:04d}{p.month:02d}" for p in full]    # ใช้ zero padding (เติมค่าศูนย์หากมีพื้นที่ว่างหลังจากใช้ int ที่มาจาก p.year และ p.month) เช่น print(f"{23:04d}") -> "0023"
 
    missing = sorted(set(full_keys) - set(keys))

    df_range = pd.DataFrame(
        [
            {
                "min_month": f"{pmin.year:04d}{pmin.month:02d}",
                "max_month": f"{pmax.year:04d}{pmax.month:02d}",
                "n_months_detected": len(keys),
                "n_months_expected_between_min_max": len(full_keys),
                "n_missing": len(missing),
            }
        ]
    )
    df_missing = pd.DataFrame({"missing_month": missing})
    return df_range, df_missing


def quick_stats(
    path: Union[str, Path],
    band: int = 1,
    zero_eps: float = 0.0,
) -> Dict[str, float]:
    """
    Compute basic raster statistics using memory-efficient block-wise reading.
    
    Analyzes pixel values without loading entire raster into memory.
    Essential for data quality assessment and outlier detection.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to raster file.
    
    band : int, optional
        Band number to analyze (1-indexed). Default: 1.
    
    zero_eps : float, optional
        Epsilon for zero detection in float rasters.
        Values with abs(value) <= zero_eps count as zero.
        Default: 0.0.
    
    Returns
    -------
    Dict[str, float]
        Statistics dictionary with keys:
        - 'pct_zero': Percentage of valid pixels that are zero
        - 'pct_nodata_or_nan': Percentage of all pixels that are nodata or NaN
        - 'min': Minimum valid value (exclude nodata/NaN)
        - 'max': Maximum valid value (exclude nodata/NaN)
        - 'n_total': Total number of pixels
    
    Notes
    -----
    Memory efficiency:
    - Reads raster in blocks using rasterio's block_windows()
    - Processes one block at a time
    - Suitable for large rasters (>GB)
    
    Zero detection:
    - Integer rasters: exact equality (value == 0)
    - Float rasters: within epsilon (abs(value) <= zero_eps)
    
    Examples
    --------
    >>> stats = quick_stats("data/raw/o_th202301_fixed.tif")
    >>> print(f"Zero pixels: {stats['pct_zero']:.1f}%")
    >>> print(f"Data range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    >>> 
    >>> # With epsilon for float comparison
    >>> stats = quick_stats("float_raster.tif", zero_eps=1e-6)
    >>> 
    >>> # Analyze specific band
    >>> stats = quick_stats("multiband.tif", band=3)
    """
    path = Path(path)

    with rasterio.open(path) as src:
        nodata = src.nodata
        dtype = src.dtypes[band - 1]
        is_float = np.issubdtype(np.dtype(dtype), np.floating)

        # ตั้งค่า counters
        n_total = 0     # นับพิกเซลทั้งหมด
        n_bad = 0       # นับพิกเซลที่ invalid (nodata หรือ NaN)
        n_zero = 0      # นับพิกเซลที่เป็นศูนย์ (หรือใกล้ศูนย์)
        vmin = None     # เก็บค่าต่ำสุดของ valid pixels
        vmax = None     # เก็บค่าสูงสุดของ valid pixels

        # Loop อ่านข้อมูลแบบ block-wise (แบ่ง raster เป็นกลุ่มพิกเซลย่อย ๆ แล้วอ่านทีละ block)
        for _, window in src.block_windows(band):
            arr = src.read(band, window=window, masked=False)   # อ่าน block ปัจจุบัน

            # สร้าง Mask ที่เก็บค่า True สำหรับพิกเซลที่มีค่า nodata หรือ NaN เพื่อที่จะไม่ต้องนำไปคำนวณ
            bad = np.zeros(arr.shape, dtype=bool)
            if nodata is not None:
                bad |= (arr == nodata)       # Mark พิกเซลที่เป็น nodata เป็นข้อมูลเสีย
            if is_float:
                bad |= np.isnan(arr)         # Mark พิกเซลที่เป็น NaN เป็นข้อมูลเสีย

            # วิเคราะห์ good pixels
            good = ~bad
            if good.any():
                a = arr[good]
                
                # 5. หาค่าต่ำสุด/สูงสุด
                mn = float(np.min(a))   # ใช้ float(np.min(a)) เพราะ np.min() อาจคืนค่าชนิดอื่น (int, float32, etc.)
                mx = float(np.max(a))
                vmin = mn if vmin is None else min(vmin, mn)
                vmax = mx if vmax is None else max(vmax, mx)

                # 6. นับ zero 
                # pixels (ต่างกันระหว่าง float vs integer)
                if is_float:
                    n_zero += int(np.sum(np.abs(a) <= float(zero_eps)))
                else:
                    n_zero += int(np.sum(a == 0))

            # 7. อัพเดท counters
            n_bad += int(np.sum(bad))
            n_pixels += arr.size

    # 8. คำนวณสถิติสุดท้าย
    pct_bad = (n_bad / n_pixels * 100.0) if n_pixels else 0.0
    pct_zero = (n_zero / (n_pixels - n_bad) * 100.0) if (n_pixels - n_bad) else 0.0

    return {
        "pct_zero": float(pct_zero),
        "pct_nodata_or_nan": float(pct_bad),
        "min": float(vmin) if vmin is not None else float("nan"),
        "max": float(vmax) if vmax is not None else float("nan"),
        "n_pixels": float(n_pixels),
    }


def run_qc(
    folder: Union[str, Path],
    pattern: str = "o_th*.tif",
    recursive: bool = False,
    baseline_month: Optional[str] = None,
    do_quick_stats: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Execute complete quality control pipeline for monthly raster time-series data.
    
    Orchestrates the entire QC workflow: file discovery, selection, metadata extraction,
    consistency checking, and coverage analysis. Designed for processing observation
    datasets with filenames like 'o_thYYYYMM.tif' or 'o_thYYYYMM_fixed.tif'.
    
    Parameters
    ----------
    folder : Union[str, Path]
        Directory containing raster files to analyze.
        Example: "data/raw/obs_monthly_tif" or Path("data/raw/obs_monthly_tif")
        
    pattern : str, optional
        Glob pattern for matching raster files (supports wildcards).
        Default: "o_th*.tif" (matches files starting with 'o_th' and ending with '.tif')
        Example alternatives: "o_th2023*.tif" (only 2023 files), "*.tif" (all TIFFs)
        
    recursive : bool, optional
        If True, search for files recursively in subdirectories.
        If False, only search in the specified folder (non-recursive).
        Default: False
        
    baseline_month : Optional[str], optional
        Month key (YYYYMM) to use as reference for consistency checks.
        If None, uses the first month in the dataset as baseline.
        Example: "202301" (January 2023)
        
    do_quick_stats : bool, optional
        If True, compute pixel statistics for each file (slower but more thorough).
        If False, skip statistics calculation (faster for initial checks).
        Default: False
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing all QC results, organized as:
        
        - 'df_meta': DataFrame of metadata for selected files
          (one row per month, includes: path, filename, month_key, variant, 
           driver, crs, width, height, count, dtype, nodata, transform, bounds, res)
          
        - 'df_incons': DataFrame of metadata inconsistencies found
          (columns: month_key, field, expected, actual, filename, path)
          
        - 'df_selection_anom': DataFrame of file selection anomalies
          (columns: month_key, issue, path)
          
        - 'df_range': DataFrame summarizing temporal coverage
          (columns: min_month, max_month, n_months_detected, 
           n_months_expected_between_min_max, n_missing)
          
        - 'df_missing': DataFrame listing missing months within range
          (columns: missing_month)
          
        - 'df_stats': DataFrame of pixel statistics (only if do_quick_stats=True)
          (columns: month_key, filename, pct_zero, pct_nodata_or_nan, min, max, n_total)
    
    Notes
    -----
    1. File selection logic (for each month):
       - If both 'raw' and 'fixed' variants exist → selects 'fixed'
       - If only one variant exists → selects that file
       - Duplicate files of same variant → flagged as anomaly
    
    2. Consistency checks compare these fields across all months:
       - crs, width, height, count, dtype, nodata, transform, bounds, res
       
    3. Missing months are calculated between min and max month_key detected.
    
    4. Statistics calculation (if enabled) uses block-wise reading to manage memory.
    
    5. The QC pipeline follows this sequence:
       1. Find all matching files → list_rasters()
       2. Select best file per month → choose_preferred_by_month()
       3. Extract metadata → scan_raster_metas()
       4. Check consistency → find_inconsistencies()
       5. Analyze coverage → compute_month_coverage()
       6. (Optional) Calculate statistics → quick_stats()
    
    Examples
    --------
    >>> # Basic QC check (fast, metadata only)
    >>> results = run_qc("data/raw/obs_monthly_tif")
    >>> print(f"Found {len(results['df_meta'])} months of data")
    >>> if not results['df_incons'].empty:
    ...     print("WARNING: Found inconsistencies!")
    ...     print(results['df_incons'][['month_key', 'field', 'expected', 'actual']])
    
    >>> # Full QC with statistics (slower)
    >>> results = run_qc(
    ...     folder="data/raw/obs_monthly_tif",
    ...     pattern="o_th2023*.tif",  # Only 2023 files
    ...     baseline_month="202301",  # Use Jan 2023 as baseline
    ...     do_quick_stats=True       # Include pixel statistics
    ... )
    >>> 
    >>> # Check coverage
    >>> coverage = results['df_range'].iloc[0]
    >>> print(f"Data range: {coverage['min_month']} to {coverage['max_month']}")
    >>> print(f"Missing months: {coverage['n_missing']}")
    >>> 
    >>> # Quick analysis of statistics
    >>> if 'df_stats' in results:
    ...     avg_zero_pct = results['df_stats']['pct_zero'].mean()
    ...     print(f"Average zero pixels: {avg_zero_pct:.1f}%")
    
    >>> # Typical notebook workflow:
    >>> from src.utils.raster_qc import run_qc, assert_qc_passed
    >>> 
    >>> # Run QC
    >>> res = run_qc("data/raw/obs_monthly_tif", do_quick_stats=False)
    >>> 
    >>> # Inspect results
    >>> print("Metadata summary:")
    >>> print(res['df_meta'][['month_key', 'filename', 'width', 'height', 'res']])
    >>> 
    >>> print("\nCoverage:")
    >>> print(res['df_range'])
    >>> 
    >>> # Hard check (optional)
    >>> try:
    ...     assert_qc_passed(
    ...         res['df_incons'],
    ...         res['df_selection_anom'],
    ...         res['df_missing'],
    ...         allow_missing_months=True
    ...     )
    ...     print("✓ QC passed")
    ... except AssertionError as e:
    ...     print(f"✗ QC failed: {e}")
    
    See Also
    --------
    assert_qc_passed : Validate QC results against acceptance criteria
    list_rasters : File discovery function
    choose_preferred_by_month : File selection logic
    scan_raster_metas : Metadata extraction
    find_inconsistencies : Consistency checking
    compute_month_coverage : Temporal analysis
    quick_stats : Pixel statistics calculation
    """
    paths = list_rasters(folder=folder, pattern=pattern, recursive=recursive)
    chosen, df_selection_anom = choose_preferred_by_month(paths)

    df_meta = scan_raster_metas(chosen)
    df_incons = find_inconsistencies(df_meta, baseline_month=baseline_month)

    df_range, df_missing = compute_month_coverage(df_meta["month_key"].tolist() if not df_meta.empty else [])

    out: Dict[str, pd.DataFrame] = {
        "df_meta": df_meta,
        "df_incons": df_incons,
        "df_selection_anom": df_selection_anom,
        "df_range": df_range,
        "df_missing": df_missing,
    }

    if do_quick_stats and (not df_meta.empty):
        stats_rows = []
        for _, r in df_meta.iterrows():
            s = quick_stats(r["path"])
            stats_rows.append(
                {
                    "month_key": r["month_key"],
                    "filename": r["filename"],
                    **s,
                }
            )
        out["df_stats"] = pd.DataFrame(stats_rows).sort_values("month_key").reset_index(drop=True)

    return out


def assert_qc_passed(
    df_incons: pd.DataFrame,
    df_selection_anom: pd.DataFrame,
    df_missing: pd.DataFrame,
    allow_missing_months: bool = False,
) -> None:
    """
    Validate QC results and raise AssertionError if acceptance criteria are not met.
    
    Acts as a hard gate in the data processing pipeline to ensure data quality 
    before proceeding to downstream analysis. Checks for three types of issues:
    
    1. Selection anomalies (duplicate files, unknown variants)
    2. Metadata inconsistencies (CRS, resolution, dimensions mismatch)
    3. Missing months in temporal coverage (optional)
    
    Parameters
    ----------
    df_incons : pd.DataFrame
        DataFrame from `find_inconsistencies()` containing metadata inconsistencies.
        Expected columns: ['month_key', 'field', 'expected', 'actual', 'filename', 'path']
        
        Example row:
        {
            'month_key': '202302',
            'field': 'width',
            'expected': 1000,
            'actual': 500,
            'filename': 'o_th202302.tif',
            'path': 'data/raw/o_th202302.tif'
        }
        
    df_selection_anom : pd.DataFrame
        DataFrame from `choose_preferred_by_month()` containing file selection anomalies.
        Expected columns: ['month_key', 'issue', 'path']
        
        Example issues:
        - 'duplicate_fixed': Multiple 'fixed' variant files for same month
        - 'duplicate_raw': Multiple 'raw' variant files for same month
        - 'unknown_variant': File variant not 'raw' or 'fixed'
        - 'unknown_month_key': Cannot extract YYYYMM from filename
        - 'no_valid_file': No valid files found for a month
        
    df_missing : pd.DataFrame
        DataFrame from `compute_month_coverage()` listing months missing from the dataset.
        Expected columns: ['missing_month']
        
        Example: {'missing_month': '202303'} for missing March 2023
        
    allow_missing_months : bool, optional
        If True, missing months do not cause assertion failure.
        If False, any missing months cause assertion failure.
        Default: False (strict mode)
        
        Use True when analyzing incomplete datasets or when data gaps are expected.
        
    Returns
    -------
    None
        Function returns nothing if QC passes.
        
    Raises
    ------
    AssertionError
        Raised when any of these conditions are met:
        
        1. `df_selection_anom` contains any rows (selection anomalies found)
        2. `df_incons` contains any rows (metadata inconsistencies found)
        3. `df_missing` contains any rows AND `allow_missing_months=False`
           (missing months found in strict mode)
        
        Error message format:
        ```
        QC FAILED:
        - Selection anomalies found: X rows
        - Inconsistent grid/metadata found: Y diffs
        - Missing months between min/max: Z months
        ```
        
    Notes
    -----
    This function is typically called immediately after `run_qc()` in analysis notebooks
    to enforce data quality gates. It's designed to fail fast and provide clear feedback
    about what needs to be fixed before proceeding.
    
    Common workflow:
    1. Run QC to identify issues
    2. Fix issues (remove duplicates, reproject files, fill missing months)
    3. Re-run QC and assert until passes
    4. Proceed with analysis
    
    The `allow_missing_months` parameter is useful for:
    - Datasets with known gaps (e.g., sensor failures, cloudy periods)
    - Preliminary analysis before complete data collection
    - Projects where temporal continuity is not critical
    
    Examples
    --------
    >>> # Example 1: Basic QC check
    >>> from src.utils.raster_qc import run_qc, assert_qc_passed
    >>> 
    >>> # Run QC pipeline
    >>> results = run_qc("data/raw/obs_monthly_tif", do_quick_stats=False)
    >>> 
    >>> # Strict check (no missing months allowed)
    >>> try:
    ...     assert_qc_passed(
    ...         df_incons=results['df_incons'],
    ...         df_selection_anom=results['df_selection_anom'],
    ...         df_missing=results['df_missing'],
    ...         allow_missing_months=False
    ...     )
    ...     print("✅ QC passed! Data is clean and complete.")
    ... except AssertionError as e:
    ...     print(f"❌ QC failed: {e}")
    ...     print("Please fix the issues above before proceeding.")
    
    >>> # Example 2: Flexible check (allow missing months)
    >>> # Useful for ongoing projects or datasets with known gaps
    >>> try:
    ...     assert_qc_passed(
    ...         results['df_incons'],
    ...         results['df_selection_anom'],
    ...         results['df_missing'],
    ...         allow_missing_months=True  # ← Missing months OK
    ...     )
    ...     print("✅ QC passed (missing months allowed).")
    ... except AssertionError as e:
    ...     print(f"❌ QC failed even with missing months allowed: {e}")
    
    >>> # Example 3: Handling specific failure cases
    >>> # If QC fails, you can inspect individual DataFrames:
    >>> if not results['df_selection_anom'].empty:
    ...     print("Selection anomalies found:")
    ...     print(results['df_selection_anom'])
    ...     
    >>> if not results['df_incons'].empty:
    ...     print("Metadata inconsistencies found:")
    ...     print(results['df_incons'][['month_key', 'field', 'expected', 'actual']])
    ...     
    >>> if not results['df_missing'].empty:
    ...     print(f"Missing {len(results['df_missing'])} months:")
    ...     print(results['df_missing'])
    
    >>> # Example 4: Using in a pipeline
    >>> def process_data_pipeline(data_folder: str):
    ...     ```
    ...     Example pipeline that enforces QC before processing.
    ...     ```
    ...     # Step 1: Quality control
    ...     qc_results = run_qc(data_folder, do_quick_stats=False)
    ...     
    ...     # Step 2: Validate QC
    ...     assert_qc_passed(
    ...         qc_results['df_incons'],
    ...         qc_results['df_selection_anom'],
    ...         qc_results['df_missing'],
    ...         allow_missing_months=True  # Project allows missing months
    ...     )
    ...     
    ...     # Step 3: Only reaches here if QC passed
    ...     print("QC passed, proceeding with analysis...")
    ...     # ... rest of processing code ...
    
    See Also
    --------
    run_qc : Generate the DataFrames needed as input
    find_inconsistencies : Function that creates df_incons
    choose_preferred_by_month : Function that creates df_selection_anom
    compute_month_coverage : Function that creates df_missing
    
    Warnings
    --------
    - This function does not check `df_stats` (from quick_stats) as it focuses on 
      structural/data integrity issues rather than data quality statistics.
    - Empty DataFrames (no rows) indicate no issues were found for that category.
    - The function performs simple existence checks only; for more complex validation
      logic, you may need to extend or replace this function.
    """
    problems = []

    if df_selection_anom is not None and not df_selection_anom.empty:
        problems.append(f"Selection anomalies found: {len(df_selection_anom)} rows")

    if df_incons is not None and not df_incons.empty:
        problems.append(f"Inconsistent grid/metadata found: {len(df_incons)} diffs")

    if (not allow_missing_months) and df_missing is not None and (not df_missing.empty):
        problems.append(f"Missing months between min/max: {len(df_missing)} months")

    if problems:
        msg = "QC FAILED:\n- " + "\n- ".join(problems)
        raise AssertionError(msg)
