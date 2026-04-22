#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path


def _ensure_writable_dir(path_str: str) -> bool:
    try:
        path = Path(path_str).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return True
    except OSError:
        return False


def configure_matplotlib(use_agg: bool = False) -> None:
    cache_root = Path(tempfile.gettempdir()) / "ae623_plot_cache"
    mpl_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg-cache"
    fontconfig_dir = xdg_dir / "fontconfig"

    mpl_env = os.environ.get("MPLCONFIGDIR")
    if not mpl_env or not _ensure_writable_dir(mpl_env):
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    xdg_env = os.environ.get("XDG_CACHE_HOME")
    if not xdg_env or not _ensure_writable_dir(xdg_env):
        xdg_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_dir)
    fontconfig_dir.mkdir(parents=True, exist_ok=True)

    if use_agg:
        import matplotlib
        matplotlib.use("Agg")
