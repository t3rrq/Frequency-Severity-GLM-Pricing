"""Load and clean French Motor TPL frequency and severity files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FREQ_NAME = "freMTPL2freq.csv"
SEV_NAME = "freMTPL2sev.csv"

USER_DOWNLOADS = Path.home() / "Downloads"

# Hugging Face mirror of the Kaggle/CASdatasets CSVs (severity is small; frequency is already local).
SEV_URL = (
    "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2sev.csv"
)


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    outputs = root / "outputs"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "raw": raw,
        "processed": processed,
        "outputs": outputs,
        "freq": raw / FREQ_NAME,
        "sev": raw / SEV_NAME,
    }


def _copy_if_needed(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    if not src.exists():
        return
    dest.write_bytes(src.read_bytes())


def ensure_raw_files(root: Path | None = None) -> dict[str, Path]:
    paths = project_paths(root)
    _copy_if_needed(USER_DOWNLOADS / FREQ_NAME, paths["freq"])
    _copy_if_needed(USER_DOWNLOADS / SEV_NAME, paths["sev"])

    if not paths["freq"].exists():
        raise FileNotFoundError(
            f"Missing {FREQ_NAME}. Place it in {paths['raw']} or {USER_DOWNLOADS}."
        )

    if not paths["sev"].exists():
        try:
            sev = pd.read_csv(SEV_URL)
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"Missing {SEV_NAME}. Download it to {paths['raw']} or {USER_DOWNLOADS}. "
                f"Tried Hugging Face mirror and failed: {exc}"
            ) from exc
        sev.to_csv(paths["sev"], index=False)

    return paths


def load_frequency(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["IDpol"] = df["IDpol"].astype("int64")
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(int)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df = df.drop_duplicates("IDpol", keep="first")
    return df


def load_severity(path: Path) -> pd.DataFrame:
    sev = pd.read_csv(path)
    sev["IDpol"] = sev["IDpol"].astype("int64")
    sev = sev.loc[sev["ClaimAmount"] > 0].copy()
    return sev
