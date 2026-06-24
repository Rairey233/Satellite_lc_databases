import dash
from dash import dcc, html, Input, Output, State
import base64
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import flask
import os
import datetime
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
import plotly.express as px
import json
from astropy.table import Table, vstack, unique, join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings
import matplotlib.cm as cm
import astropy.units as u
from astropy.time import Time
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from flask_github import GitHub
plt.style.use('default')
from spacetrack import SpaceTrackClient
import json
from skyfield.api import Topos, load, EarthSatellite
from skyfield.api import N,S,E,W, wgs84, Loader
from sgp4 import omm
from sgp4.api import Satrec
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
loader = Loader('/var/tmp/')
timescale = loader.timescale()
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from skyfield import almanac
from pytz import timezone
from astroplan import Observer
from astropy.coordinates import EarthLocation, AltAz

def login(username,password):
    # Initialize SpaceTrackClient with your credentials
    st = SpaceTrackClient('Robert.Airey@warwick.ac.uk','BellaCube138!!!')
    return st

st = login('username','password')

# All GEO objects that have been updated in the last 2 weeks
targets = st.generic_request('gp', mean_motion='0.99--1.01', epoch='>now-14')

with open('geo.json', 'w') as outfile:
    json.dump(targets, outfile, indent=2)


DISCOSWEB_API_URL = os.getenv("DISCOSWEB_API_URL", "https://discosweb.esoc.esa.int/api").rstrip("/")
DISCOSWEB_TOKEN = os.getenv("DISCOSWEB_TOKEN", "IjUzODhlMzQ0LTZmMTgtNGVmYS1hNjIyLTJlNzVmOGI4N2Y4NSI.U4aEnaj5mpU5SAYhR43Gp19_g1o").strip()
DISCOSWEB_TIMEOUT = float(os.getenv("DISCOSWEB_TIMEOUT", "10"))


def normalize_norad(value: Any) -> Optional[str]:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return str(int(digits))


def discos_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.api+json",
        "DiscosWeb-Api-Version": "2",
        "User-Agent": "STING-Observations/1.0",
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
    dims: List[str] = []
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
            [html.H4("DISCOSweb Metadata"), html.Div("No DISCOSweb data available.")],
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
                html.H4("DISCOSweb Metadata"),
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
            html.H4("DISCOSweb Metadata"),
            html.Table(rows, style={"width": "100%"}),
        ],
        style={
            "padding": "0.75rem",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
        },
    )


def selected_object_card(selected_row: Dict[str, Any]) -> html.Div:
    if not selected_row:
        return html.Div(
            [html.H4("Selected Object"), html.Div("Select a row to view DISCOSweb metadata.")],
            style={
                "padding": "0.75rem",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fafafa",
            },
        )

    fields = [
        ("Name", selected_row.get("Name")),
        ("NORAD ID", selected_row.get("NORAD ID")),
        ("Type", selected_row.get("Type")),
        ("Date Observed", selected_row.get("DATE OBSERVED")),
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
            html.H4("Selected Object"),
            html.Table(rows, style={"width": "100%"}),
        ],
        style={
            "padding": "0.75rem",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
        },
    )


def make_altitude_plot(object_ids,time_span):
    # Define the location of La Palma
    la_palma = Topos('28.7606 N', '17.8795 W')

    # Set the start time to midnight of the current day
    start_time = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Set the end time to one and a half months from the start time
    end_time = start_time + datetime.timedelta(days=time_span)

    # Generate the time range with 1-hour intervals
    time_interval = datetime.timedelta(hours=24)
    times = np.arange(start_time, end_time, time_interval)

    # Convert numpy.datetime64 to datetime
    times = Time([t.item() for t in times])

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)


    with open('geo.json', 'r') as infile:
        targets = json.load(infile)

    
    for t in targets:
        object_id = t['NORAD_CAT_ID']
        if int(object_id) != object_ids:
            continue

        name = t['OBJECT_NAME']
        sat = Satrec()
        omm.initialize(sat, t)
        satellite = EarthSatellite.from_satrec(sat, timescale)

        # Calculate the altitude for the satellite at each time step
        difference = satellite - la_palma
        topocentric = difference.at(timescale.from_astropy(times))
        alt, _, _ = topocentric.altaz()
        altitude_data = alt.degrees

    return times.datetime,altitude_data

def get_visibility_tonight(object_ids):
    
    # Define the observer location
    latitude = 28.7606
    longitude = -17.8795
    elevation = 2350.0
    location_pipe = EarthLocation(lat=latitude, lon=longitude, height=elevation)

    eph = load('de421.bsp')
    earth, moon,sun = eph['earth'], eph['moon'],eph['sun']

    location = Topos(latitude_degrees= 28.7606,longitude_degrees= -17.8795,elevation_m=2350.0)

    # Set the start time to midnight of the current day
    start_time = datetime.datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)

    # Set the end time to one and a half months from the start time
    end_time = start_time + datetime.timedelta(days=1)

    # Generate the time range with 1-hour intervals
    time_interval = datetime.timedelta(hours=1)
    times = np.arange(start_time, end_time, time_interval)

    # Convert numpy.datetime64 to datetime
    times = Time([t.item() for t in times])

    with open('geo.json', 'r') as infile:
        targets = json.load(infile)

    for t in targets:
        object_id = t['NORAD_CAT_ID']
        if int(object_id) != object_ids:
            continue

        name = t['OBJECT_NAME']
        sat = Satrec()
        omm.initialize(sat, t)
        satellite = EarthSatellite.from_satrec(sat, timescale)

        difference = satellite - location
        topocentric = difference.at(timescale.from_astropy(times))
        alt, _, _ = topocentric.altaz()
        altitude_data = alt.degrees
        
        moon_topocentric = (earth + location).at(timescale.from_astropy(times)).observe(moon)
        moon_alt, moon_az, _ = moon_topocentric.apparent().altaz()
        moon_altitude_data = moon_alt.degrees
        
        # Create an observer object
        observer = Observer(location=location_pipe, name="Custom Observer", timezone="UTC")
        
        time = times[0]
        observer.time = time
        sun_altaz = observer.sun_altaz(time)
        moon_altaz = observer.moon_altaz(time)
        # Twilight definitions
        astro_twilight_start = observer.twilight_evening_astronomical(time, which='next')
        astro_twilight_end = observer.twilight_morning_astronomical(time, which='next')
        
        sunrise_time = observer.sun_rise_time(time, which='next')
        sunset_time = observer.sun_set_time(time, which='next')
        

    return times.datetime,altitude_data,moon_altitude_data,sunrise_time.datetime,sunset_time.datetime,astro_twilight_start.datetime,astro_twilight_end.datetime
    

def illuminated(info, times, atmospheric_height=0*u.km):
    """
    Return a boolean mask indicating whether the target
    is fully illuminated by the sun at given times
    :param time: Astropy time(s) to evaluate
    :param atmospheric_height: Effective height of the atmosphere to consider in the shadow cone
    :return: Numpy boolean array with points fully illuminated by the sun set to True
    """
    from skyfield.sgp4lib import EarthSatellite
    from sgp4 import omm
    from skyfield.api import Loader
    from sgp4.api import Satrec
    import astropy.constants.iau2015 as const
    loader = Loader('/var/tmp/')
    timescale = loader.timescale()
    de421 = loader('de421.bsp')
    
    sat = Satrec()
    omm.initialize(sat, info)
    satellite = EarthSatellite.from_satrec(sat, timescale)

    # Calculate the distance to and angular separation between the earth and sun
    # from the perspective of the satellite
    internal_time = timescale.from_astropy(times)
    sun = de421['sun'].at(internal_time)
    earth = de421['earth'].at(internal_time)
    sat = (de421['earth'] + satellite).at(internal_time)
    earth_sun_angle = (earth - sat).separation_from(sun - sat).to(u.degree)
    sat_earth_distance = (earth - sat).distance().to(u.km)
    sat_sun_distance = (sun - sat).distance().to(u.km)

    # Project the earth-sun separation and apparent solar size to physical distances
    # on the sun-satellite tangent plane that intersects the center of the earth
    sun_angular_radius = np.arcsin(const.R_sun / sat_sun_distance)
    projected_sun_radius = np.tan(sun_angular_radius) * sat_earth_distance * np.cos(earth_sun_angle)
    projected_separation = sat_earth_distance * np.sin(earth_sun_angle)

    # Satellite is fully illuminated if the separation is greater than the effective size of earth + sun
    return projected_separation > projected_sun_radius + const.R_earth + atmospheric_height



def lc_bin(time, flux, bin_width):
    '''
    Function to bin the data into bins of a given width. time and bin_width 
    must have the same units
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        edges = np.arange(np.min(time), np.max(time), bin_width)
        dig = np.digitize(time, edges)
        time_binned = (edges[1:] + edges[:-1]) / 2
        flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
        err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else sem(flux[dig == i]) for i in range(1, len(edges))])
        time_bin = time_binned[~np.isnan(err_binned)]
        flux_bin = flux_binned[~np.isnan(err_binned)]

    return time_bin, flux_bin

def load_data(name, object_id, radius, blendthreshold=0.05):
    data = Table.read(name)
    data = data[data['object'] == object_id]
    ref_flux = np.min(np.vstack([data['ref_a_flux_' + str(radius)], data['ref_b_flux_' + str(radius)]]), axis=0)
    blendfrac = ref_flux / (data['flux_' + str(radius)] + ref_flux)
    return data[blendfrac < blendthreshold]

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

def calculate_midnight_altitude(norad_id):
        # Define the location of La Palma
    la_palma = Topos('28.7606 N', '17.8795 W')

    # Set the start time to midnight of the current day
    start_time = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)



    with open('geo.json', 'r') as infile:
        targets = json.load(infile)

    
    for t in targets:
        object_id = t['NORAD_CAT_ID']
        if int(object_id) != norad_id:
            continue

        name = t['OBJECT_NAME']
        sat = Satrec()
        omm.initialize(sat, t)
        satellite = EarthSatellite.from_satrec(sat, timescale)

        # Calculate the altitude for the satellite at each time step
        difference = satellite - la_palma
        topocentric = difference.at(timescale.from_astropy(Time(start_time)))
        alt, _, _ = topocentric.altaz()
        altitude_data = alt.degrees
        return altitude_data
    


# Read data from the CSV file
data_file = 'observations.csv'
df = pd.read_csv(data_file)
# Read the image file
with open("Logo.png", "rb") as img_file:
    encoded_image = base64.b64encode(img_file.read()).decode('ascii')

# Calculate the altitude at midnight for each row and filter based on this value
df['Midnight Altitude'] = df['NORAD ID'].apply(calculate_midnight_altitude)

# Filter the DataFrame for rows where the altitude is greater than 30 degrees
visible_objects_df = df[df['Midnight Altitude'] > 30]

# Remove duplicates based on the 'NORAD ID' to get unique objects
unique_visible_objects_df = visible_objects_df.drop_duplicates(subset=['NORAD ID'])

# Select only the necessary columns for the visible objects table
visible_objects_filtered_df = unique_visible_objects_df[['Name', 'NORAD ID', 'Type']]

app.layout = html.Div(
    children=[
        # Common parent element for logo and title
        html.Div(
            children=[
                # Logo
                html.Img(
                    src=f"data:image/png;base64,{encoded_image}",
                    style={'height': '75px', 'width': 'auto', 'vertical-align': 'middle', 'margin-right': '20px'}
                ),
                # Title
                html.H1('STING Observations', style={'display': 'inline-block', 'vertical-align': 'middle'})
            ],
            style={'text-align': 'center', 'line-height': '50px'}  # Set line-height equal to the logo height
        ),
        # Add the new component for displaying unique satellite count here
        html.Div([
            html.H3("Database Statistics"),
            html.P(id='unique-satellites-count')
        ], style={'text-align': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
        # Table displaying satellite observation data with checkboxes
        dash_table.DataTable(
            id='table',
            columns=[{'name': col, 'id': col} for col in df.columns if col != 'Midnight Altitude'],
            data=df.to_dict('records'),
            row_selectable='multi',
            selected_rows=[],
            filter_action='native',
            filter_query='',
            style_table={'height': '300px', 'overflowY': 'auto'},
            filter_options={
                'case': 'insensitive',  # Make the search case-insensitive
                'operator': 'contains'  # Use the 'contains' operator for filtering
            },
            sort_action='native',
            sort_mode='multi'
        ),
        # Download button
        html.Button('Download CSV', id='download-button'),
        
        # Plot button
        html.Button('Show Light-Curve Plot', id='plot-button'),
        
        # Add a button to trigger the generation of the visibility plot
        html.Button('Show Visibility Plot', id='visibility-plot-button'),
        
        html.Button('Show Visibility Tonight Plot', id='visibility-tonight-plot-button'),
        
        # Hidden divs to store selected rows and download links
        html.Div(id='selected-rows', style={'display': 'none'}),
        html.Div(id='download-links', style={'display': 'none'}),

        html.Div(
            [
                html.Div(id='selected-object-card'),
                html.Div(
                    [
                        html.H3('DISCOSweb Enrichment'),
                        html.Div(
                            'DISCOSweb live enrichment enabled.' if DISCOSWEB_TOKEN else 'DISCOSweb live enrichment disabled (set DISCOSWEB_TOKEN).',
                            id='discosweb-status',
                            style={'margin-bottom': '10px'}
                        ),
                        html.Div(id='discosweb-panel')
                    ]
                ),
            ],
            style={
                'display': 'grid',
                'gridTemplateColumns': 'minmax(250px, 1fr) minmax(400px, 2fr)',
                'gap': '20px',
                'margin-top': '20px',
                'margin-bottom': '20px'
            }
        ),
        # New table for unique visible objects
        html.H3('Unique Visible Objects (Altitude > 30° at Midnight)'),
        dash_table.DataTable(
            id='visible-objects-table',
            columns=[{'name': col, 'id': col} for col in visible_objects_filtered_df.columns],
            data=visible_objects_filtered_df.to_dict('records'),
            style_table={'height': '200px', 'overflowY': 'auto'},
            filter_action='native',
            filter_query='',
            filter_options={
                'case': 'insensitive',  # Make the search case-insensitive
                'operator': 'contains'  # Use the 'contains' operator for filtering
            },
            sort_action='native',
            sort_mode='multi'
        ),
        # Download button and component
        html.Button("Download Visible Objects List", id="download_vis_list"),
        dcc.Download(id="download-dataframe-csv"),
        
        # Placeholder for the light-curve plot
        dcc.Graph(id='light-curve-plot'),

        # Placeholder for the visibility plot
        dcc.Graph(id='visibility-plot'),

        # Placeholder for the visibility tonight plot
        dcc.Graph(id='visibility-tonight-plot'),
        # Add this new component for the top 10 satellites table
        html.Div([
            html.H3("Top 10 Most Observed Satellites"),
            dash_table.DataTable(
                id='top-satellites-table',
                columns=[
                    {"name": "Satellite Name", "id": "name"},
                    {"name": "Observation Count", "id": "count"}
                ],
                style_table={'overflowX': 'auto'},
            )
        ])
    ]
)


# Define callbacks to handle interactions with the app
@app.callback(
    Output('selected-rows', 'children'),
    Input('table', 'selected_rows')
)
def store_selected_rows(selected_rows):
    return json.dumps(selected_rows)


@app.callback(
    Output('selected-object-card', 'children'),
    Output('discosweb-status', 'children'),
    Output('discosweb-panel', 'children'),
    Input('table', 'derived_virtual_data'),
    Input('table', 'derived_virtual_selected_rows')
)
def update_discosweb_metadata(table_data, selected_rows):
    visible_rows = table_data if table_data is not None else df.to_dict('records')
    status_text = (
        'DISCOSweb live enrichment enabled.'
        if DISCOSWEB_TOKEN
        else 'DISCOSweb live enrichment disabled (set DISCOSWEB_TOKEN).'
    )

    if not visible_rows:
        empty_state = html.Div('No observations available.')
        return empty_state, status_text, discos_card({})

    if not selected_rows:
        selected_row = visible_rows[0]
    else:
        selected_index = selected_rows[0]
        selected_row = visible_rows[selected_index] if selected_index < len(visible_rows) else visible_rows[0]

    norad_id = selected_row.get('NORAD ID')
    metadata = discos_lookup(norad_id)
    return selected_object_card(selected_row), status_text, discos_card(metadata)

@app.callback(
    Output('download-button', 'disabled'),
    Input('selected-rows', 'children')
)
def enable_download_button(selected_rows):
    return False if selected_rows else True

@app.callback(
    Output('download-links', 'children'),
    Input('download-button', 'n_clicks'),
    State('selected-rows', 'children')
)

def generate_download_links(n_clicks, selected_rows):
    if not n_clicks:
        return []

    selected_rows = json.loads(selected_rows)
    download_links = []

    for row_idx in selected_rows:
        selected_row = df.iloc[row_idx]
        norad_id = selected_row['NORAD ID']
        observation_date = selected_row['DATE OBSERVED']
        csv_filename = f'CSV/{observation_date}.csv'

        if os.path.exists(csv_filename):
            csv_data = pd.read_csv(csv_filename)
            csv_data = csv_data[csv_data['object'] == norad_id]

            filtered_csv_filename = f'Filtered_{observation_date}_{norad_id}.csv'
            csv_data.to_csv(filtered_csv_filename, index=False)

            download_link = html.A('Download CSV', href=f'/download/{filtered_csv_filename}', target='_blank')
            download_links.append(download_link)

    return download_links

def load_data(name, object_id, radius, blendthreshold=0.05):
    data = Table.read(name)
    data = data[data['object'] == object_id]
    ref_flux = np.min(np.vstack([data['ref_a_flux_' + str(radius)], data['ref_b_flux_' + str(radius)]]), axis=0)
    blendfrac = ref_flux / (data['flux_' + str(radius)] + ref_flux)
    return data[blendfrac < blendthreshold]

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download_vis_list", "n_clicks"),
    prevent_initial_call=True,
)
def download_visible_objects(n_clicks):
    # Convert the filtered DataFrame to a CSV string
    csv_string = visible_objects_filtered_df.to_csv(index=False)
    
    # Create a downloadable file with the CSV string
    return dcc.send_data_frame(visible_objects_filtered_df.to_csv, "visible_objects.csv", index=False)

@app.callback(
    Output('light-curve-plot', 'figure'),
    Input('plot-button', 'n_clicks'),
    State('selected-rows', 'children')
)
def generate_light_curve_plot(n_clicks, selected_rows):
    if not n_clicks:
        return {}
    # Create subplots with shared x-axes
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    selected_rows = json.loads(selected_rows)
    for row_idx in selected_rows:
        radius = 4
        bin_width = None
        selected_row = df.iloc[row_idx]
        norad_id = selected_row['NORAD ID']
        name = selected_row['Name']
        b_tag = selected_row['B']
        g_tag = selected_row['G']
        r_tag = selected_row['R']
        i_tag = selected_row['I']
        observation_date = selected_row['DATE OBSERVED']
        data = load_data('CSV/{}.csv'.format(observation_date), norad_id, radius)

        
        with open('VIS/{}.vis'.format(observation_date), 'r') as infile:
            info = json.load(infile)
        
        object_info = None
        for i in info:
            if i['NORAD_CAT_ID'] == str(norad_id):
                object_info = i
                break

        data = data[illuminated(object_info, Time(data['utc']))]
        data_b = data[data['camera_id'] == 'CAM1']
        data_g = data[data['camera_id'] == 'CAM2']
        data_r = data[data['camera_id'] == 'CAM3']
        data_i = data[data['camera_id'] == 'CAM4']
        
        phase_r = data_r['phase']
        phase_g = data_g['phase']
        phase_b = data_b['phase']
        phase_i = data_i['phase']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mag_r = data_r['zeropoint_mean'] - 2.5 * np.log10(data_r['flux_' + str(radius)])
            mag_g = data_g['zeropoint_mean'] - 2.5 * np.log10(data_g['flux_' + str(radius)])
            mag_b = data_b['zeropoint_mean'] - 2.5 * np.log10(data_b['flux_' + str(radius)])
            mag_i = data_i['zeropoint_mean'] - 2.5 * np.log10(data_i['flux_' + str(radius)])
        if len(data_b) > 0 and len(data_i) >0:
            data_bminusi = join(data_b, data_i,'frame')
            phase_bmi = data_bminusi['phase_1']
            mag_bmi_b = data_bminusi['zeropoint_mean_1'] - 2.5 * np.log10(data_bminusi['flux_' + str(radius) + '_1'])
            mag_bmi_i = data_bminusi['zeropoint_mean_2'] - 2.5 * np.log10(data_bminusi['flux_' + str(radius) + '_2'])
            mag_bmi = mag_bmi_b - mag_bmi_i
        # Create a figure with subplots
        ms = 5

        # Add scatter plot traces to the subplots
        if b_tag == 'Y':
            scatter_zp_b = go.Scatter(x=phase_b, y=data_b['zeropoint_mean'] ,mode='markers', marker=dict(color='darkblue', size=ms),name = 'ZP<sub>B</sub>')
            scatter_b = go.Scatter(x=phase_b, y=mag_b,mode='markers', marker=dict(color='blue', size=ms), name='B<sub>RGB</sub>')
            fig.add_trace(scatter_b, row=3, col=1)
            fig.add_trace(scatter_zp_b,row=1,col = 1)

        if g_tag == 'Y':
            scatter_zp_g = go.Scatter(x=phase_g, y=data_g['zeropoint_mean'] ,mode='markers', marker=dict(color='darkgreen', size=ms),name = 'ZP<sub>G</sub>')
            scatter_g = go.Scatter(x=phase_g, y=mag_g,mode='markers',  marker=dict(color='green', size=ms), name='G<sub>RGB</sub>')
            fig.add_trace(scatter_g, row=3, col=1)
            fig.add_trace(scatter_zp_g,row=1,col = 1)

        if r_tag == 'Y':
            scatter_zp_r = go.Scatter(x=phase_r, y=data_r['zeropoint_mean'] ,mode='markers', marker=dict(color='darkred', size=ms),name = 'ZP<sub>R</sub>')
            scatter_r = go.Scatter(x=phase_r, y=mag_r,mode='markers',  marker=dict(color='red', size=ms), name='R<sub>RGB</sub>')
            fig.add_trace(scatter_zp_r,row=1,col = 1)
            fig.add_trace(scatter_r, row=3, col=1)

        if i_tag == 'Y':
            scatter_zp_i = go.Scatter(x=phase_i, y=data_i['zeropoint_mean'] ,mode='markers', marker=dict(color='sandybrown', size=ms),name = 'ZP<sub>i</sub>')
            scatter_i = go.Scatter(x=phase_i, y=mag_i,mode='markers',  marker=dict(color='brown', size=ms), name='i<sub>sdss</sub>')
            fig.add_trace(scatter_zp_i,row=1,col = 1)
            fig.add_trace(scatter_i, row=3, col=1)


        if b_tag == 'Y' and i_tag == 'Y':
            fig.add_trace(go.Scatter(x=phase_bmi, y=mag_bmi, mode='markers', marker=dict(color='#FF00FF', size=ms), name='Colour Index'), row=2, col=1)

        # Update subplot titles and axis labels
        fig.update_yaxes(title_text = 'Zeropoint-mean [MAG]',row = 1 , col = 1,tickangle = 0)
        fig.update_yaxes(title_text="Colour Index\n(B - i) [MAG]", row=2, col=1,tickangle = 0)
        fig.update_xaxes(showticklabels=False, row=2, col=1)
        fig.update_yaxes(title_text="Corrected Photometry [MAG]", row=3, col=1,tickangle = 0)
        fig.update_xaxes(title_text="Solar Eq. Phase Angle (Degrees)", row=3, col=1)

        # Align y-axes
        #fig.update_yaxes(matches='y', row=1, col=1)
        #fig.update_yaxes(matches='y', row=2, col=1)
        #fig.update_yaxes(matches='y', row=3, col=1)

        # Adjust title text size
        title_font_size = 10  # Change this to your preferred font size

        fig.update_yaxes(title_font=dict(size=title_font_size), row=1, col=1)
        fig.update_xaxes(title_font=dict(size=title_font_size), row=2, col=1)
        fig.update_yaxes(title_font=dict(size=title_font_size), row=2, col=1)
        fig.update_xaxes(title_font=dict(size=title_font_size), row=3, col=1)
        fig.update_yaxes(title_font=dict(size=title_font_size), row=3, col=1)

        # Adjust margins to accommodate larger labels
        fig.update_layout(margin=dict(l=100, r=20, t=50, b=50))
        
        fig.update_yaxes(range=[15,22], row=1, col=1)
        fig.update_yaxes(range=[-1,3], row=2, col=1)


        # Show the legend
        fig.update_layout(title=str(name) + ' - ' + str(observation_date))
        fig.update_layout(showlegend=True)
        fig.update_layout(legend= {'itemsizing': 'constant'})
        fig.update_yaxes(autorange="reversed", row=3, col=1)
        fig.update_layout(font=dict(family="Times New Roman"))
        

        return fig


@app.callback(
    Output('visibility-plot', 'figure'),
    Input('visibility-plot-button', 'n_clicks'),
    State('selected-rows', 'children')
)
def generate_visibility_plot(n_clicks, selected_rows):
    if not n_clicks:
        return {}

    selected_rows = json.loads(selected_rows)

    for row_idx in selected_rows:
        selected_row = df.iloc[row_idx]
        norad_id = selected_row['NORAD ID']
        name = selected_row['Name']
        time_data,altitude_data = make_altitude_plot(object_ids=norad_id, time_span=60)

        # Create a Plotly figure using the altitude data
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Scatter(x=time_data, y=altitude_data, mode='lines',name = str(name)))
        
        # Update layout of the figure
        fig.update_layout(
            title="Visibility Plot",
            xaxis_title="DateTime (UTC)",
            yaxis_title="Altitude (degrees)",
            showlegend=True,
            legend_title="Satellite NORAD ID",
            height=600,
            width=800
        )

    return fig

@app.callback(
    Output('visibility-tonight-plot', 'figure'),
    Input('visibility-tonight-plot-button', 'n_clicks'),
    State('selected-rows', 'children')
)

def generate_visibility_tonight_plot(n_clicks, selected_rows):
    if not n_clicks:
        return {}

    selected_rows = json.loads(selected_rows)


    for row_idx in selected_rows:
        selected_row = df.iloc[row_idx]
        norad_id = selected_row['NORAD ID']
        name = selected_row['Name']
        
        # Call the function to get visibility data for tonight
        time_data, altitude_data,moon_altitude_data,sunrise_time_data,sunset_time_data,twi_start_data,twi_end_data = get_visibility_tonight(norad_id)
        
        # Create a Plotly figure using the altitude data
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Scatter(x=time_data, y=altitude_data, mode='lines',name = str(name) + ' ' +'Altitude'))
        
        
        # Add moon altitude trace
        fig.add_trace(go.Scatter(x=time_data, y=moon_altitude_data, mode='lines', name='Moon Altitude', line=dict(color='grey', dash='dash')), row=1, col=1)
        
        # Add vertical lines for sunset, sunrise, and twilight
        fig.add_vline(x=sunset_time_data, line=dict(color='orange', dash='dash'), row=1, col=1)
        fig.add_vrect(x0=twi_start_data, x1=twi_end_data, fillcolor='darkgrey', opacity=0.2, line=dict(width=0), row=1, col=1)
        fig.add_vline(x=sunrise_time_data, line=dict(color='red', dash='dash'), row=1, col=1)

        # Add invisible scatter traces for custom legend entries
        fig.add_scatter(x=[None], y=[None], mode='markers', marker=dict(size=0, color='orange'), name=f"Sunset Time: {sunset_time_data}")
        fig.add_scatter(x=[None], y=[None], mode='markers', marker=dict(size=0, color='darkgrey'), name=f"Astronomical Twilight Start: {twi_start_data}")
        fig.add_scatter(x=[None], y=[None], mode='markers', marker=dict(size=0, color='darkgrey'), name=f"Astronomical Twilight End: {twi_end_data}")
        fig.add_scatter(x=[None], y=[None], mode='markers', marker=dict(size=0, color='red'), name=f"Sunrise Time: {sunrise_time_data}")
        
        # Set x-axis tick format and rotation
        fig.update_xaxes(tickformat='%H:%M:%S', tickangle=90, row=1, col=1)
        
        # Set x-axis and y-axis labels
        fig.update_xaxes(title_text='DateTime (UTC)', row=1, col=1)
        fig.update_yaxes(title_text='Altitude (degrees)', row=1, col=1)
        
        # Set legend
        fig.update_layout(legend_title='Legend')
        
        # Set y-axis range
        fig.update_yaxes(range=[30, 90], row=1, col=1)
        
        # Set layout title
        fig.update_layout(title='Visibility Tonight Plot')

        # Update layout for dark background
        fig.update_layout(
            plot_bgcolor='black',  # Set background color
            paper_bgcolor='black',  # Set plot area color
            font=dict(color='white')  # Set font color
        )

        print(f"Sunset Time: {sunset_time_data}")
        print(f"Sunrise Time: {sunrise_time_data}")
        
        print(f"Astronomical Twilight Start: {twi_start_data}")
        print(f"Astronomical Twilight End: {twi_end_data}")


    return fig
@app.callback(
    Output('top-satellites-table', 'data'),
    Input('table', 'data')
)
def update_top_satellites_table(table_data):
    # Convert the table data to a DataFrame
    df = pd.DataFrame(table_data)
    
    # Count the occurrences of each satellite name
    satellite_counts = df['Name'].value_counts().reset_index()
    satellite_counts.columns = ['name', 'count']
    
    # Get the top 10 most common satellites
    top_10_satellites = satellite_counts.head(10).to_dict('records')
    
    return top_10_satellites

@app.callback(
    Output('unique-satellites-count', 'children'),
    Input('table', 'data')
)
def update_unique_satellites_count(table_data):
    df = pd.DataFrame(table_data)
    
    # Drop duplicates based on 'NORAD ID' to ensure unique objects
    unique_objects_df = df.drop_duplicates(subset=['NORAD ID'])
    
    # Get counts of each type for unique objects
    type_counts = unique_objects_df['Type'].value_counts()
    
    # Extract counts for each type
    act_count = type_counts.get('ACT', 0)
    rb_count = type_counts.get('RB', 0)
    def_count = type_counts.get('DEF', 0)
    deb_count = type_counts.get('DEB', 0)
    
    # Calculate unique and total counts
    unique_count = unique_objects_df['NORAD ID'].nunique()
    total_count = len(df['NORAD ID'])
    # Filter for 'ACT' and 'DEF' types
    filtered_df = unique_objects_df[unique_objects_df['Type'].isin(['ACT', 'DEF'])]
    
    # Count the bus configurations for 'ACT' and 'DEF' types
    bus_counts = filtered_df['BUS'].value_counts().nlargest(10)
    
    # Create a bar chart for bus configurations
    bus_fig = px.bar(
        x=bus_counts.index,
        y=bus_counts.values,
        labels={'x': 'Bus Configuration', 'y': 'Count'},
    )
    
    # Update the bar colors and add black edge color for the bus chart
    bus_fig.update_traces(marker_color='lightblue', marker_line_color='black', marker_line_width=1.5)

    # Create a bar chart using Plotly
    fig = px.bar(
        x=['ACTIVE SATS', 'DEFUNCT SATS', 'Rocket Bodies', 'Debris'],
        y=[act_count, def_count, rb_count, deb_count],
        labels={'x': 'Type', 'y': 'Count'},
    )
    # Update the bar colors and add black edge color
    fig.update_traces(marker_color=['blue', 'green', 'red', 'orange'],
                      marker_line_color='black',
                      marker_line_width=1.5)
    
    return [
        "Number of Unique Objects: ",
        html.Span(f"{unique_count}", style={'font-weight': 'bold', 'color': 'DarkMagenta'}),
        " and Total Number of Observations: ",
        html.Span(f"{total_count}", style={'font-weight': 'bold', 'color': 'purple'}),
        html.Br(),
        "ACTIVE SATS: ", html.Span(f"{act_count}", style={'font-weight': 'bold', 'color': 'blue'}),
        html.Br(),
        "DEFUNCT SATS: ", html.Span(f"{def_count}", style={'font-weight': 'bold', 'color': 'green'}),
        html.Br(),
        "Rocket Bodies: ", html.Span(f"{rb_count}", style={'font-weight': 'bold', 'color': 'red'}),
        html.Br(),
        "Debris: ", html.Span(f"{deb_count}", style={'font-weight': 'bold', 'color': 'orange'}),
        # Add the bar chart
        dcc.Graph(figure=bus_fig, style={'position': 'absolute', 'top': '0', 'left': '0', 'width': '30%', 'height': '30%'}),
        dcc.Graph(figure=fig, style={'position': 'absolute', 'top': '0', 'right': '0', 'width': '30%', 'height': '30%'})
    ]

if __name__ == '__main__':
    app.run_server(port=8051)
