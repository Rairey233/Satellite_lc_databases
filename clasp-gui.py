from __future__ import annotations

import os
import re
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html, dash_table
from plotly.subplots import make_subplots


CLASP_ROOT = Path(os.getenv("CLASP_ROOT", "/Users/robertairey/CLASP"))
TLE_PATH = Path(os.getenv("CLASP_TLE_PATH", "/Users/robertairey/CLASP/ALL_OBJECTS.txt"))
PORT = int(os.getenv("PORT", "8050"))

DISCOSWEB_API_URL = os.getenv("DISCOSWEB_API_URL", "https://discosweb.esoc.esa.int/api").rstrip("/")
DISCOSWEB_TOKEN = os.getenv("DISCOSWEB_TOKEN", "").strip()
DISCOSWEB_TIMEOUT = float(os.getenv("DISCOSWEB_TIMEOUT", "10"))


def resolve_currentcat_path() -> Path:
    candidates = [
        os.getenv("CURRENTCAT_PATH"),
        os.getenv("CLASP_CURRENTCAT_PATH"),
        str(CLASP_ROOT / "currentcat.tsv"),
        str(Path.cwd() / "currentcat.tsv"),
        "/Users/robertairey/CLASP/currentcat.tsv",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(candidates[0] or str(CLASP_ROOT / "currentcat.tsv"))


CURRENTCAT_PATH = resolve_currentcat_path()


# -----------------------------------------------------------------------------
# Local catalogue build
# -----------------------------------------------------------------------------
def normalize_norad(value: Any) -> Optional[str]:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return str(int(digits))


def load_tle_dict(tle_path: Path) -> Dict[str, str]:
    if not tle_path.exists():
        return {}

    with open(tle_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    tle_dict: Dict[str, str] = {}
    for i in range(0, len(lines) - 2, 3):
        object_name = lines[i].strip()
        if object_name.startswith("0 "):
            object_name = object_name[2:]

        line1 = lines[i + 1]
        norad_raw = line1[2:7].strip() if len(line1) >= 7 else ""
        norad_id = normalize_norad(norad_raw)
        if norad_id:
            tle_dict[norad_id] = object_name

    return tle_dict


def parse_sensor_file(file_path: Path) -> Optional[Dict[str, Any]]:
    filename = file_path.name
    if filename.endswith("_adjusted_optical_data.csv"):
        sensor = "OPTICAL"
    elif filename.endswith("_adjusted_swir_data.csv"):
        sensor = "SWIR"
    else:
        return None

    parts = filename.split("_")
    if len(parts) < 7:
        return None

    return {
        "Date Observed": parts[0],
        "NORAD ID": normalize_norad(parts[1]),
        "cluster_number": parts[2],
        "cluster_id": parts[3],
        "Sensor": sensor,
        "Filename": filename,
        "File Path": str(file_path),
    }


def build_catalogue(root_directory: Path, tle_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for file_path in root_directory.rglob("*.csv"):
        parsed = parse_sensor_file(file_path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Object Name",
                "NORAD ID",
                "Date Observed",
                "cluster_number",
                "cluster_id",
                "OPTICAL",
                "SWIR",
                "OPTICAL Filename",
                "SWIR Filename",
                "OPTICAL Path",
                "SWIR Path",
            ]
        )

    tle_dict = load_tle_dict(tle_path)
    all_files = pd.DataFrame(rows)
    all_files["Object Name"] = all_files["NORAD ID"].map(tle_dict).fillna("Unknown")

    grouped = (
        all_files.groupby(
            ["Object Name", "NORAD ID", "Date Observed", "cluster_number", "cluster_id", "Sensor"],
            as_index=False,
        )
        .agg(
            {
                "Filename": lambda values: " | ".join(sorted(set(values))),
                "File Path": lambda values: " | ".join(sorted(set(values))),
            }
        )
    )

    index_cols = ["Object Name", "NORAD ID", "Date Observed", "cluster_number", "cluster_id"]

    flags = (
        grouped.assign(flag="Y")
        .pivot_table(index=index_cols, columns="Sensor", values="flag", aggfunc="first", fill_value="N")
        .reset_index()
    )

    filenames = (
        grouped.pivot_table(index=index_cols, columns="Sensor", values="Filename", aggfunc="first")
        .reset_index()
        .rename(columns={"OPTICAL": "OPTICAL Filename", "SWIR": "SWIR Filename"})
    )

    paths = (
        grouped.pivot_table(index=index_cols, columns="Sensor", values="File Path", aggfunc="first")
        .reset_index()
        .rename(columns={"OPTICAL": "OPTICAL Path", "SWIR": "SWIR Path"})
    )

    catalogue = flags.merge(filenames, on=index_cols, how="left").merge(paths, on=index_cols, how="left")

    for col in ["OPTICAL", "SWIR"]:
        if col not in catalogue.columns:
            catalogue[col] = "N"
    for col in ["OPTICAL Filename", "SWIR Filename", "OPTICAL Path", "SWIR Path"]:
        if col not in catalogue.columns:
            catalogue[col] = ""

    catalogue[["OPTICAL Filename", "SWIR Filename", "OPTICAL Path", "SWIR Path"]] = catalogue[
        ["OPTICAL Filename", "SWIR Filename", "OPTICAL Path", "SWIR Path"]
    ].fillna("")

    catalogue = catalogue.sort_values(
        ["Date Observed", "NORAD ID", "cluster_number", "cluster_id"]
    ).reset_index(drop=True)
    return catalogue


def load_currentcat(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return pd.DataFrame()

    header = lines[0].rstrip("\n")
    if header.startswith("#"):
        header = header[1:]

    body_lines = [header + "\n"]
    for line in lines[1:]:
        if line.startswith("# Updated"):
            continue
        if not line.strip():
            continue
        body_lines.append(line)

    try:
        currentcat = pd.read_csv(StringIO("".join(body_lines)), sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()

    currentcat.columns = [str(c).strip() for c in currentcat.columns]
    if "Satcat" not in currentcat.columns:
        return pd.DataFrame()

    currentcat["NORAD ID"] = currentcat["Satcat"].apply(normalize_norad)
    currentcat = currentcat[currentcat["NORAD ID"].notna()].copy()
    currentcat = currentcat.drop_duplicates(subset=["NORAD ID"], keep="first")
    return currentcat


def currentcat_lookup(currentcat: pd.DataFrame, norad_id: Any) -> Dict[str, Any]:
    norad_id = normalize_norad(norad_id)
    if currentcat.empty or not norad_id:
        return {}

    match = currentcat.loc[currentcat["NORAD ID"] == norad_id]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def currentcat_card(metadata: Dict[str, Any]) -> html.Div:
    if not metadata:
        return html.Div(
            [html.H4("Current catalogue metadata"), html.Div("No currentcat.tsv record found for this object.")],
            style={
                "padding": "0.75rem",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fafafa",
            },
        )

    fields = [
        ("Name", metadata.get("Name")),
        ("NORAD ID", metadata.get("NORAD ID")),
        ("JCAT", metadata.get("JCAT")),
        ("Piece", metadata.get("Piece")),
        ("Active", metadata.get("Active")),
        ("Type", metadata.get("Type")),
        ("Parent", metadata.get("Parent")),
        ("Owner", metadata.get("Owner")),
        ("State", metadata.get("State")),
        ("Launch Date", metadata.get("LDate")),
        ("Start Date", metadata.get("SDate")),
        ("Status", metadata.get("ExpandedStatus")),
        ("Decay Date", metadata.get("DDate")),
        ("Other Date", metadata.get("ODate")),
        ("Period", metadata.get("Period")),
        ("Perigee", metadata.get("Perigee")),
        ("Apogee", metadata.get("Apogee")),
        ("Inclination", metadata.get("Inc")),
        ("Operational Orbit", metadata.get("OpOrbit")),
    ]

    rows = []
    for label, value in fields:
        if value in (None, "", "-", [], {}):
            continue
        rows.append(
            html.Tr(
                [
                    html.Th(label, style={"textAlign": "left", "paddingRight": "1rem", "verticalAlign": "top"}),
                    html.Td(str(value)),
                ]
            )
        )

    return html.Div(
        [
            html.H4("Current catalogue metadata"),
            html.Table(rows, style={"width": "100%"}),
        ],
        style={
            "padding": "0.75rem",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
        },
    )


# -----------------------------------------------------------------------------
# DISCOSweb 
# -----------------------------------------------------------------------------
def discos_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.api+json",
        "DiscosWeb-Api-Version": "2",
        "User-Agent": "CLASP-LightCurve-Browser/1.0",
    }
    if DISCOSWEB_TOKEN:
        headers["Authorization"] = f"Bearer {DISCOSWEB_TOKEN}"
    return headers


def format_bool(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return ""


def format_dimensions(attrs: Dict[str, Any]) -> str:
    dims = []
    for key, label in [
        ("width", "W"),
        ("height", "H"),
        ("depth", "D"),
        ("diameter", "Dia"),
        ("span", "Span"),
    ]:
        value = attrs.get(key)
        if value not in (None, "", "-"):
            dims.append(f"{label}: {value}")
    return ", ".join(dims)


def resolve_relationship_items(
    rel: Optional[Dict[str, Any]],
    included_lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not rel or "data" not in rel:
        return []

    data = rel["data"]
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]

    out: List[Dict[str, Any]] = []
    for item in data:
        key = (str(item.get("type", "")), str(item.get("id", "")))
        resolved = included_lookup.get(key)
        if resolved:
            out.append(resolved)
    return out


def names_from_resources(resources: List[Dict[str, Any]]) -> str:
    names: List[str] = []
    for resource in resources:
        attrs = resource.get("attributes", {}) or {}
        name = attrs.get("name")
        if name not in (None, "", "-"):
            names.append(str(name))
    return ", ".join(sorted(set(names)))


def first_resource(resources: List[Dict[str, Any]]) -> Dict[str, Any]:
    return resources[0] if resources else {}


def extract_error_detail(doc: Dict[str, Any]) -> str:
    errors = doc.get("errors")
    if not isinstance(errors, list) or not errors:
        return ""
    first = errors[0] or {}
    detail = first.get("detail") or first.get("title") or first.get("code")
    return str(detail) if detail else ""


@lru_cache(maxsize=1024)
def discos_lookup(norad_id: Any) -> Dict[str, Any]:
    norad_id = normalize_norad(norad_id)
    if not norad_id:
        return {}

    if not DISCOSWEB_TOKEN:
        return {
            "_status": "disabled",
            "_message": "DISCOSweb not configured. Set DISCOSWEB_TOKEN to enable live enrichment.",
        }

    params = {
        "filter": f"eq(satno,{norad_id})",
        "include": "launch,launch.site,launch.vehicle,reentry,states,operators",
        "page[size]": 1,
    }

    try:
        response = requests.get(
            f"{DISCOSWEB_API_URL}/objects",
            headers=discos_headers(),
            params=params,
            timeout=DISCOSWEB_TIMEOUT,
        )
    except requests.RequestException as exc:
        return {
            "_status": "error",
            "_message": f"DISCOSweb lookup failed: {exc}",
        }

    try:
        doc = response.json()
    except Exception:
        doc = {}

    if not response.ok:
        detail = extract_error_detail(doc)
        message = f"DISCOSweb lookup failed with HTTP {response.status_code}"
        if detail:
            message += f": {detail}"
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            message += f" (Retry-After: {retry_after}s)"
        return {
            "_status": "error",
            "_message": message,
        }

    data = doc.get("data") or []
    if not data:
        return {
            "_status": "not_found",
            "_message": f"No DISCOSweb object found for NORAD {norad_id}.",
        }

    obj = data[0]
    attrs = obj.get("attributes", {}) or {}
    rels = obj.get("relationships", {}) or {}

    included = doc.get("included") or []
    included_lookup = {
        (str(item.get("type", "")), str(item.get("id", ""))): item
        for item in included
        if item.get("type") is not None and item.get("id") is not None
    }

    launch_items = resolve_relationship_items(rels.get("launch"), included_lookup)
    reentry_items = resolve_relationship_items(rels.get("reentry"), included_lookup)
    state_items = resolve_relationship_items(rels.get("states"), included_lookup)
    operator_items = resolve_relationship_items(rels.get("operators"), included_lookup)

    launch_resource = first_resource(launch_items)
    reentry_resource = first_resource(reentry_items)

    launch_attrs = launch_resource.get("attributes", {}) or {}
    reentry_attrs = reentry_resource.get("attributes", {}) or {}
    launch_rels = launch_resource.get("relationships", {}) or {}

    vehicle_items = resolve_relationship_items(launch_rels.get("vehicle"), included_lookup)
    site_items = resolve_relationship_items(launch_rels.get("site"), included_lookup)

    return {
        "_status": "ok",
        "Name": attrs.get("name"),
        "NORAD ID": attrs.get("satno"),
        "COSPAR ID": attrs.get("cosparId"),
        "Object Class": attrs.get("objectClass"),
        "Mass (kg)": attrs.get("mass"),
        "Shape": attrs.get("shape"),
        "Dimensions (m)": format_dimensions(attrs),
        "Max Cross Section (m²)": attrs.get("xSectMax"),
        "Min Cross Section (m²)": attrs.get("xSectMin"),
        "Avg Cross Section (m²)": attrs.get("xSectAvg"),
        "Mission": attrs.get("mission"),
        "Active": format_bool(attrs.get("active")),
        "First Epoch": attrs.get("firstEpoch"),
        "Predicted Decay": attrs.get("predDecayDate"),
        "Launch Epoch": launch_attrs.get("epoch"),
        "Launch No": launch_attrs.get("cosparLaunchNo"),
        "Flight No": launch_attrs.get("flightNo"),
        "Launch Failure": format_bool(launch_attrs.get("failure")),
        "Launch Site": names_from_resources(site_items),
        "Launch Vehicle": names_from_resources(vehicle_items),
        "States": names_from_resources(state_items),
        "Operators": names_from_resources(operator_items),
        "Reentry Epoch": reentry_attrs.get("epoch"),
    }


def discos_card(metadata: Dict[str, Any]) -> html.Div:
    status = metadata.get("_status")

    if not metadata:
        return html.Div(
            [html.H4("DISCOSweb metadata"), html.Div("No DISCOSweb data available.")],
            style={
                "padding": "0.75rem",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fafafa",
            },
        )

    if status in {"disabled", "error", "not_found"}:
        return html.Div(
            [
                html.H4("DISCOSweb metadata"),
                html.Div(metadata.get("_message", "Unavailable")),
            ],
            style={
                "padding": "0.75rem",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fafafa",
            },
        )

    rows = []
    for label, value in metadata.items():
        if label.startswith("_"):
            continue
        if value in (None, "", "-", [], {}):
            continue
        rows.append(
            html.Tr(
                [
                    html.Th(label, style={"textAlign": "left", "paddingRight": "1rem", "verticalAlign": "top"}),
                    html.Td(str(value)),
                ]
            )
        )

    return html.Div(
        [
            html.H4("DISCOSweb metadata"),
            html.Table(rows, style={"width": "100%"}),
        ],
        style={
            "padding": "0.75rem",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
        },
    )


# -----------------------------------------------------------------------------
# Light-curve reading and plotting
# -----------------------------------------------------------------------------
def split_first_path(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    path_string = str(value).split(" | ")[0].strip()
    return Path(path_string) if path_string else None


def load_lightcurve_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["datetime", "time", "magnitude"])

    df = pd.read_csv(path)
    mag_col = next((c for c in df.columns if str(c).strip().lower() in {"# magnitude", "magnitude", "mag"}), None)
    dt_col = next((c for c in df.columns if str(c).strip().lower() in {"datetime", "utc", "time_utc"}), None)
    rel_col = next((c for c in df.columns if str(c).strip().lower() in {"time", "seconds", "elapsed_time"}), None)

    out = pd.DataFrame()
    out["magnitude"] = pd.to_numeric(df[mag_col], errors="coerce") if mag_col else pd.Series(dtype=float)
    out["datetime"] = pd.to_datetime(df[dt_col], errors="coerce") if dt_col else pd.Series(dtype="datetime64[ns]")
    out["time"] = pd.to_numeric(df[rel_col], errors="coerce") if rel_col else pd.Series(dtype=float)

    out = out.dropna(subset=["magnitude"], how="all").copy()

    if out["time"].isna().all() and out["datetime"].notna().any():
        out["time"] = (out["datetime"] - out["datetime"].min()).dt.total_seconds()

    return out


def add_lightcurve_trace(fig: go.Figure, df: pd.DataFrame, row: int, title: str, use_datetime: bool) -> None:
    if df.empty:
        fig.add_annotation(
            text=f"No {title.lower()} data",
            xref=f"x{row}" if row > 1 else "x",
            yref=f"y{row}" if row > 1 else "y",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return

    use_dt = use_datetime and df["datetime"].notna().any()
    x = df["datetime"] if use_dt else df["time"]
    x_title = "Datetime" if use_dt else "Elapsed time (s)"

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=df["magnitude"],
            mode="lines+markers",
            name=title,
            marker={"size": 5},
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="Magnitude", autorange="reversed", row=row, col=1)
    fig.update_xaxes(title_text=x_title, row=row, col=1)


def make_lightcurve_figure(row_data: Dict[str, Any], use_datetime: bool = False) -> go.Figure:
    optical_df = load_lightcurve_csv(split_first_path(row_data.get("OPTICAL Path")))
    swir_df = load_lightcurve_csv(split_first_path(row_data.get("SWIR Path")))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=("Optical light curve", "SWIR light curve"),
    )

    add_lightcurve_trace(fig, optical_df, row=1, title="Optical", use_datetime=use_datetime)
    add_lightcurve_trace(fig, swir_df, row=2, title="SWIR", use_datetime=use_datetime)

    fig.update_layout(
        height=750,
        title=(
            f"{row_data.get('Object Name', 'Unknown')} | NORAD {row_data.get('NORAD ID', '')} | "
            f"{row_data.get('Date Observed', '')} | {row_data.get('cluster_number', '')} {row_data.get('cluster_id', '')}"
        ),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 70, "r": 20, "t": 80, "b": 50},
    )
    return fig


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def build_app(catalogue: pd.DataFrame, currentcat: pd.DataFrame) -> Dash:
    app = Dash(__name__)

    table_columns = [
        "Object Name",
        "NORAD ID",
        "Date Observed",
        "cluster_number",
        "cluster_id",
        "OPTICAL",
        "SWIR",
        "OPTICAL Filename",
        "SWIR Filename",
    ]

    currentcat_status = (
        f"Loaded currentcat.tsv from {CURRENTCAT_PATH} with {len(currentcat):,} rows."
        if not currentcat.empty
        else f"currentcat.tsv not loaded from {CURRENTCAT_PATH}."
    )

    discos_status = (
        "DISCOSweb live enrichment enabled."
        if DISCOSWEB_TOKEN
        else "DISCOSweb live enrichment disabled (set DISCOSWEB_TOKEN)."
    )

    app.layout = html.Div(
        [
            html.H1("CLASP Light-Curve Browser"),
            html.P(
                "Browse observations, view optical/SWIR light curves, and inspect "
                "object metadata from currentcat.tsv and DISCOSweb."
            ),
            html.Div(
                [
                    html.Div(currentcat_status, style={"fontWeight": "bold"}),
                    html.Div(discos_status, style={"marginTop": "0.25rem"}),
                    dcc.Checklist(
                        id="time-axis-mode",
                        options=[{"label": "Use datetime on x-axis", "value": "datetime"}],
                        value=[],
                        style={"marginTop": "0.5rem"},
                    ),
                ],
                style={
                    "padding": "1rem",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "backgroundColor": "#fafafa",
                    "marginBottom": "1rem",
                },
            ),
            dcc.Store(id="catalogue-store", data=catalogue.to_dict("records")),
            dash_table.DataTable(
                id="catalogue-table",
                columns=[{"name": c, "id": c} for c in table_columns],
                data=catalogue[table_columns].to_dict("records"),
                row_selectable="single",
                selected_rows=[0] if not catalogue.empty else [],
                page_size=15,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "6px", "whiteSpace": "normal", "height": "auto"},
                style_data_conditional=[
                    {"if": {"filter_query": "{OPTICAL} = Y", "column_id": "OPTICAL"}, "fontWeight": "bold"},
                    {"if": {"filter_query": "{SWIR} = Y", "column_id": "SWIR"}, "fontWeight": "bold"},
                ],
            ),
            html.Div(
                [
                    html.Div(id="selected-row-summary"),
                    html.Div(id="currentcat-panel"),
                    html.Div(id="discos-panel"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                    "gap": "1rem",
                    "marginTop": "1rem",
                },
            ),
            dcc.Graph(id="lightcurve-graph", style={"marginTop": "1rem"}),
        ],
        style={"maxWidth": "1800px", "margin": "0 auto", "padding": "1rem"},
    )

    @app.callback(
        Output("selected-row-summary", "children"),
        Output("currentcat-panel", "children"),
        Output("discos-panel", "children"),
        Output("lightcurve-graph", "figure"),
        Input("catalogue-table", "derived_virtual_data"),
        Input("catalogue-table", "derived_virtual_selected_rows"),
        Input("time-axis-mode", "value"),
        State("catalogue-store", "data"),
        prevent_initial_call=False,
    )
    def update_selection(
        visible_rows: Optional[List[Dict[str, Any]]],
        selected_rows: Optional[List[int]],
        time_axis_mode: List[str],
        catalogue_records: List[Dict[str, Any]],
    ):
        visible_base_rows = visible_rows if visible_rows else catalogue_records
        if not visible_base_rows:
            return html.Div("No CLASP light-curve files found."), html.Div(), html.Div(), go.Figure()

        if not selected_rows:
            selected_visible_row = visible_base_rows[0]
        else:
            selected_index = selected_rows[0]
            selected_visible_row = (
                visible_base_rows[selected_index]
                if selected_index < len(visible_base_rows)
                else visible_base_rows[0]
            )

        match_keys = ["Object Name", "NORAD ID", "Date Observed", "cluster_number", "cluster_id"]

        row_data = next(
            (
                record
                for record in catalogue_records
                if all(str(record.get(k, "")) == str(selected_visible_row.get(k, "")) for k in match_keys)
            ),
            selected_visible_row,
        )

        currentcat_metadata = currentcat_lookup(currentcat, row_data.get("NORAD ID"))
        discos_metadata = discos_lookup(row_data.get("NORAD ID"))

        summary = html.Div(
            [
                html.H4("Selected observation"),
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Th("Object Name", style={"textAlign": "left", "paddingRight": "1rem"}),
                                html.Td(row_data.get("Object Name", "")),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Th("NORAD ID", style={"textAlign": "left", "paddingRight": "1rem"}),
                                html.Td(row_data.get("NORAD ID", "")),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Th("Date Observed", style={"textAlign": "left", "paddingRight": "1rem"}),
                                html.Td(row_data.get("Date Observed", "")),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Th("Cluster", style={"textAlign": "left", "paddingRight": "1rem"}),
                                html.Td(f"{row_data.get('cluster_number', '')} / {row_data.get('cluster_id', '')}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Th("Optical file", style={"textAlign": "left", "paddingRight": "1rem"}),
                                html.Td(row_data.get("OPTICAL Filename", "")),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Th("SWIR file", style={"textAlign": "left", "paddingRight": "1rem"}),
                                html.Td(row_data.get("SWIR Filename", "")),
                            ]
                        ),
                    ]
                ),
            ],
            style={
                "padding": "0.75rem",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fafafa",
            },
        )

        figure = make_lightcurve_figure(row_data, use_datetime=("datetime" in time_axis_mode))
        return summary, currentcat_card(currentcat_metadata), discos_card(discos_metadata), figure

    return app


if __name__ == "__main__":
    catalogue_df = build_catalogue(CLASP_ROOT, TLE_PATH)
    currentcat_df = load_currentcat(CURRENTCAT_PATH)
    app = build_app(catalogue_df, currentcat_df)
    app.run(debug=True, port=PORT)