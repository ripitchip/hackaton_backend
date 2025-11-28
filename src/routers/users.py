from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependency; fall back to an in-memory list if pandas is not installed
try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # ImportError or other import issues
    pd = None  # type: ignore
    np = None  # type: ignore

router = APIRouter(prefix="/users", tags=["users"])


# Candidate data files (checked in order). Update this list if your assets live elsewhere.
CANDIDATE_FILES = [
    Path("assets/users.xlsx"),
    Path("assets/users.csv"),
    Path("assets/ERP_Equipes Airplus.xlsx"),
    Path("assets/people.xlsx"),
]

MES_CANDIDATES = [
    Path("assets/MES_Extraction.xlsx"),
    Path("assets/MES_Extraction.csv"),
    Path("assets/MES.xlsx"),
]


def _fallback_df():
    # Minimal fallback when pandas isn't available or no file found
    return [{"username": "Rick"}, {"username": "Morty"}]


def _find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _duration_to_hours(series):
    if pd is None:
        return pd.Series(dtype=float)  # type: ignore
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    td = pd.to_timedelta(text, errors="coerce")
    hours = td.dt.total_seconds() / 3600
    numeric = pd.to_numeric(text.str.replace(",", "."), errors="coerce")
    return hours.fillna(numeric)


def _has_alea(value) -> bool:
    if pd is None:
        return False
    if pd.isna(value):
        return False
    s = str(value).strip().upper()
    return s not in {"", "0", "RAS", "OK", "NONE", "NAN", "AUCUN", "AUCUNE", "-"}


def get_employee_stats(employee_name: str) -> Dict[str, Any] | None:
    """Compute simple per-employee stats using ERP + MES assets.

    Returns a JSON-serializable dict or None if not computable.
    """
    if pd is None:
        return None

    erp_path = _find_first_existing(CANDIDATE_FILES)
    mes_path = _find_first_existing(MES_CANDIDATES)
    if not erp_path or not mes_path:
        return None

    try:
        if erp_path.suffix.lower() in {".xlsx", ".xls"}:
            erp_df = pd.read_excel(erp_path)
        else:
            erp_df = pd.read_csv(erp_path)
    except Exception:
        return None

    try:
        if mes_path.suffix.lower() in {".xlsx", ".xls"}:
            mes_df = pd.read_excel(mes_path)
        else:
            mes_df = pd.read_csv(mes_path)
    except Exception:
        return None

    # try to find prenom/nom columns in ERP
    cols = [c for c in erp_df.columns]
    cols_lower = {c.lower(): c for c in cols}
    first_candidates = (
        "first_name",
        "firstname",
        "prenom",
        "prénom",
        "given_name",
        "givenname",
    )
    last_candidates = (
        "last_name",
        "lastname",
        "nom",
        "surname",
    )
    first_col = next(
        (cols_lower[c] for c in first_candidates if c in cols_lower),
        None,
    )
    last_col = next(
        (cols_lower[c] for c in last_candidates if c in cols_lower),
        None,
    )

    if not first_col or not last_col:
        # fallback to username-like column
        username_col = next(
            (cols_lower[c] for c in ("username", "name") if c in cols_lower),
            None,
        )
        if not username_col:
            return None
        erp_df["full_name"] = erp_df[username_col].astype(str)
    else:
        erp_df["full_name"] = (
            erp_df[first_col].astype(str).str.strip()
            + " "
            + erp_df[last_col].astype(str).str.strip()
        ).str.strip()

    # normalize and search
    erp_df["_full_norm"] = erp_df["full_name"].astype(str).str.lower().str.strip()
    target = employee_name.strip().lower()
    matches = erp_df[erp_df["_full_norm"] == target]
    if matches.empty:
        # partial match
        matches = erp_df[erp_df["_full_norm"].str.contains(target, na=False)]
    if matches.empty:
        return None

    emp = matches.iloc[0]

    # attempt to get assigned poste and parse digits
    poste_val = None
    for c in erp_df.columns:
        if c.lower().startswith("poste") or c.lower() == "poste":
            poste_val = emp.get(c)
            break

    poste_num = None
    if poste_val is not None:
        import re

        nums = re.findall(r"\d+", str(poste_val))
        if nums:
            poste_num = nums[0]

    # compute MES stats if poste_num found
    if poste_num is None:
        return None

    mes_cols = [c for c in mes_df.columns]
    # try to find cycle time column
    temps_reel = next(
        (c for c in mes_cols if "temps" in c.lower() and "r" in c.lower()),
        None,
    )
    # detect alea column
    alea_col = next((c for c in mes_cols if "al" in c.lower()), None)

    # coerce Poste column to string and filter
    poste_col = next((c for c in mes_cols if c.lower().startswith("poste")), None)
    if not poste_col:
        return None
    mes_df["_poste_str"] = mes_df[poste_col].astype(str).str.strip()
    poste_data = (
        mes_df[mes_df["_poste_str"].str.contains(str(poste_num), na=False)].copy()
    )
    if poste_data.empty:
        return None

    # compute cycle_time
    if temps_reel:
        poste_data["cycle_time"] = _duration_to_hours(poste_data[temps_reel])
    else:
        poste_data["cycle_time"] = pd.NA

    # aleas
    if alea_col:
        poste_data["has_alea"] = poste_data[alea_col].apply(_has_alea)
    else:
        poste_data["has_alea"] = False

    nb_ops = len(poste_data)
    cycle_mean = None
    if nb_ops > 0 and "cycle_time" in poste_data:
        try:
            cycle_mean = float(poste_data["cycle_time"].dropna().mean())
        except Exception:
            cycle_mean = None

    nb_aleas = int(poste_data["has_alea"].sum()) if nb_ops > 0 else 0
    taux_aleas = round((nb_aleas / nb_ops * 100), 1) if nb_ops > 0 else 0.0

    # performance score (simple): combine ecart (if exists) and aleas
    ecart_mean = None
    # try to find an ecart column
    ecart_col = next((c for c in mes_cols if "ecart" in c.lower()), None)
    if ecart_col and ecart_col in poste_data:
        try:
            ecart_mean = float(poste_data[ecart_col].dropna().mean())
        except Exception:
            ecart_mean = None

    ecart_score = 100
    if ecart_mean is not None:
        ecart_score = max(0, 100 - abs(ecart_mean) * 100)
    alea_score = max(0, 100 - taux_aleas)
    perf_score = round((ecart_score * 0.6 + alea_score * 0.4), 1)

    stats = {
        "nb_operations": nb_ops,
        "cycle_moyen_heures": round(cycle_mean, 3) if cycle_mean is not None else None,
        "ecart_moyen_heures": round(ecart_mean, 3) if ecart_mean is not None else None,
        "nb_aleas": nb_aleas,
        "taux_aleas_pct": taux_aleas,
        "performance_score": perf_score,
    }

    return stats


@lru_cache()
def load_users_df() -> List[Dict[str, Any]]:
    """Load users from the first existing file in CANDIDATE_FILES.

    Returns a list of dict rows. If pandas is not installed or no file found,
    returns a small fallback list.
    """
    if pd is None:
        return _fallback_df()

    for p in CANDIDATE_FILES:
        if p.exists():
            try:
                if p.suffix.lower() in {".xlsx", ".xls"}:
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
            except Exception:
                continue

            if df.empty:
                continue

            # Normalize column lookup
            cols_lower = {c.lower(): c for c in df.columns}

            # Detect first / last name columns to build a full name
            first_candidates = (
                "first_name",
                "firstname",
                "given_name",
                "givenname",
                "prenom",
                "prénom",
                "first",
            )
            last_candidates = (
                "last_name",
                "lastname",
                "family_name",
                "familyname",
                "nom",
                "surname",
                "last",
            )

            first_col = next(
                (cols_lower[c] for c in first_candidates if c in cols_lower),
                None,
            )
            last_col = next(
                (cols_lower[c] for c in last_candidates if c in cols_lower),
                None,
            )

            if first_col and last_col:
                df["first_name"] = df[first_col].astype(str).str.strip()
                df["last_name"] = df[last_col].astype(str).str.strip()
                df["full_name"] = (
                    df["first_name"].fillna("")
                    + " "
                    + df["last_name"].fillna("")
                ).str.strip()

            # Try to find a username-like column (fallbacks)
            for candidate in ("username", "name", "nom", "login"):
                if candidate in cols_lower:
                    df = df.rename(columns={cols_lower[candidate]: "username"})
                    break

            # If we constructed a full_name, use it as the username to return
            # a "Prenom Nom" style value
            if "full_name" in df.columns:
                df["username"] = df["full_name"]

            # If still no username column, use the first column
            if "username" not in df.columns:
                df = df.rename(columns={df.columns[0]: "username"})

            # Normalize usernames and convert rows to dicts
            df["username"] = df["username"].astype(str).str.strip()
            rows = df.to_dict(orient="records")
            return rows

    return _fallback_df()


@router.get("/")
async def read_users():
    """Return the list of usernames (from Excel/CSV if available)."""
    rows = load_users_df()
    # rows might be list of dicts; ensure we return only usernames
    usernames = []
    for r in rows:
        if isinstance(r, dict) and r.get("username"):
            usernames.append({"username": str(r.get("username"))})
    return usernames


@router.get("/{username}")
async def read_user(username: str):
    """Return the full data row for the given username (case-insensitive)."""
    rows = load_users_df()
    # rows may be list of dicts; perform case-insensitive match
    for r in rows:
        if not isinstance(r, dict):
            continue
        val = r.get("username")
        if val is None:
            continue
        if str(val).strip().lower() == username.strip().lower():
            # return only non-null fields and include computed stats when available
            user = {k: v for k, v in r.items() if v is not None}
            stats = None
            try:
                stats = get_employee_stats(user.get("username", ""))
            except Exception:
                stats = None
            return {"user": user, "stats": stats}

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
