
import re
import itertools
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import streamlit as st

# Country coordinates are stored locally: no external geodata package required.

# NetworkX is optional: used for the co-authorship network
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="What's up Science?",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

JOURNAL_URL = (
    "https://raw.githubusercontent.com/louloupM/what-s-up-science/"
    "main/Scientific%20journal%20list%201.2.csv"
)

COLORS = [
    "#536DFE", "#00A6A6", "#FF9F43", "#8B5CF6",
    "#E85D75", "#2F9E72", "#4C78A8", "#F2C14E"
]

STOPWORDS = set("""
a an the and or but if while with without from into onto upon about
above below between through during before after over under again
further then once here there when where why how all any both each
few more most other some such no nor not only own same so than too
very can will just should now is are was were be been being have
has had do does did of to in on for by as at this that these those
it its we our their they them he she his her you your study studies
results result using used use based novel new high low different
effect effects method methods approach approaches analysis
analyses characterization investigation development application
applications properties property activity activities role paper
show shows showed demonstrate demonstrated may might could also
however therefore within across among first second one two three
four five data significant
""".split())

COUNTRY_ALIASES = {
    "usa": "USA",
    "u.s.a.": "USA",
    "us": "USA",
    "united states": "USA",
    "uk": "GBR",
    "u.k.": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "south korea": "KOR",
    "north korea": "PRK",
    "russia": "RUS",
    "iran": "IRN",
    "vietnam": "VNM",
    "taiwan": "TWN",
    "czech republic": "CZE",
    "the netherlands": "NLD"
}


# ============================================================
# 2. MODERN INTERFACE
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1550px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #E8EDF4;
}

.hero {
    padding: 1.7rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    background: linear-gradient(
        115deg, #172554 0%, #273C91 55%, #536DFE 100%
    );
    color: white;
    box-shadow: 0 12px 30px rgba(35,55,120,.15);
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 .3rem 0;
}

.hero p {
    color: #DBE5FF;
    margin: 0;
    font-size: .98rem;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #172033;
    margin: .3rem 0 .15rem;
}

.section-sub {
    color: #718096;
    font-size: .88rem;
    margin-bottom: .9rem;
}

.kpi {
    border: 1px solid #E8EDF4;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    background: linear-gradient(145deg, #FFFFFF, #F8FAFF);
    min-height: 108px;
}

.kpi-label {
    font-size: .76rem;
    color: #718096;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.kpi-value {
    font-family: 'Manrope', sans-serif;
    font-size: 1.8rem;
    color: #172033;
    font-weight: 800;
    margin-top: .25rem;
}

.kpi-note {
    font-size: .76rem;
    color: #8A96A8;
    margin-top: .1rem;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E8EDF4;
    padding: 14px 16px;
    border-radius: 15px;
}

div[data-testid="stTabs"] button {
    font-weight: 650;
}

.stDownloadButton button {
    border-radius: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E8EDF4;
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔬 What's up Science?</h1>
    <p>
        Scientific publication analytics · Bibliometrics ·
        Research themes · International collaborations
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATA IMPORT AND SCOPUS NORMALIZATION
# ============================================================

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes):

    errors = []

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding,
                sep=None,
                engine="python"
            )
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("Unable to read CSV: " + " | ".join(errors))


def normalize_column(name):
    return re.sub(
        r"[\s_\-]+", " ", str(name).strip().lower()
    )


def find_column(df, candidates):

    lookup = {
        normalize_column(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column(candidate)

        if key in lookup:
            return lookup[key]

    return None


@st.cache_data(show_spinner=False)
def prepare_data(file_bytes):

    raw = read_uploaded_csv(file_bytes)
    raw.columns = [str(c).strip() for c in raw.columns]

    # Scopus columns and alternative naming conventions
    aliases = {
        "Title": ["Title", "Article title"],
        "Year": ["Year", "Publication Year"],
        "Journal": ["Source title", "Journal", "Source Title"],
        "Cited by": ["Cited by", "Citations", "Times Cited"],
        "DOI": ["DOI"],
        "Authors": ["Authors", "Author(s)"],
        "Author full names": [
            "Author full names", "Authors full names"
        ],
        "Author(s) ID": ["Author(s) ID", "Author ID"],
        "Affiliations": [
            "Affiliations", "Author Affiliations"
        ],
        "Authors with affiliations": [
            "Authors with affiliations"
        ],
        "Abstract": ["Abstract"],
        "Author Keywords": [
            "Author Keywords", "Author keywords"
        ],
        "Index Keywords": [
            "Index Keywords", "Indexed Keywords"
        ],
        "Publisher": ["Publisher"],
        "Document Type": ["Document Type", "Type"],
        "Open Access": ["Open Access"],
        "EID": ["EID"],
        "Link": ["Link"],
        "ISSN": ["ISSN"],
        "Language": [
            "Language of Original Document", "Language"
        ],
        "Publication Stage": ["Publication Stage"]
    }

    mapped = {}

    for target, candidates in aliases.items():
        source = find_column(raw, candidates)

        if source is not None:
            mapped[target] = raw[source]
        else:
            mapped[target] = np.nan

    required = ["Title", "Year", "Journal"]

    missing = [
        name for name in required
        if pd.isna(mapped[name]).all()
    ]

    if missing:
        raise ValueError(
            "Missing required Scopus columns: "
            + ", ".join(missing)
            + ". Expected Title, Year and Source title."
        )

    df = pd.DataFrame(mapped)

    # Preserve all original Scopus columns
    for col in raw.columns:
        if col not in df.columns:
            df[col] = raw[col]

    text_columns = [
        "Title", "Journal", "Authors",
        "Author full names", "Author(s) ID",
        "Affiliations", "Authors with affiliations",
        "Abstract", "Author Keywords", "Index Keywords",
        "Publisher", "Document Type", "Open Access",
        "DOI", "EID", "Link"
    ]

    for col in text_columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["Year"] = pd.to_numeric(
        df["Year"], errors="coerce"
    )

    df["Cited by"] = pd.to_numeric(
        df["Cited by"], errors="coerce"
    ).fillna(0)

    df["Journal"] = df["Journal"].replace(
        "", "Unknown journal"
    )

    df["Document Type"] = df["Document Type"].replace(
        "", "Unknown"
    )

    df["Publisher"] = df["Publisher"].replace(
        "", "Unknown"
    )

    df["DOI URL"] = df["DOI"].apply(
        lambda x: (
            x if x.startswith("http")
            else "https://doi.org/" + x if x
            else ""
        )
    )

    # Combined text for thematic analysis
    df["Text for topics"] = (
        df["Title"] + " "
        + df["Abstract"] + " "
        + df["Author Keywords"] + " "
        + df["Index Keywords"]
    )

    return df, raw.columns.tolist()


# ============================================================
# 4. OPTIONAL JOURNAL DATABASE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_journal_database():

    try:
        journals = pd.read_csv(JOURNAL_URL)

        journals.columns = journals.columns.str.strip()

        if len(journals.columns) < 5 or "Title" not in journals.columns:
            return pd.DataFrame(
                columns=["Journal_key", "Publisher", "Domain"]
            )

        publisher_col = journals.columns[3]
        domain_col = journals.columns[4]

        result = journals[
            ["Title", publisher_col, domain_col]
        ].copy()

        result.columns = [
            "Journal", "Publisher", "Domain"
        ]

        result["Journal_key"] = (
            result["Journal"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return result[
            ["Journal_key", "Publisher", "Domain"]
        ].drop_duplicates("Journal_key")

    except Exception:
        return pd.DataFrame(
            columns=["Journal_key", "Publisher", "Domain"]
        )


# ============================================================
# 5. TEXT AND KEYWORD PROCESSING
# ============================================================

def split_scopus(value):

    if not value:
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def tokenize(text):

    text = str(text).lower()
    text = text.replace("–", " ").replace("—", " ")

    words = re.findall(
        r"[a-z][a-z-]{2,}", text
    )

    return [
        word for word in words
        if word not in STOPWORDS
    ]


def keyword_counter(frame, column):

    counts = Counter()

    for value in frame[column].fillna(""):
        counts.update(
            item.lower().strip()
            for item in split_scopus(value)
            if len(item.strip()) > 2
        )

    return counts


def text_counter(frame, column):

    counts = Counter()

    for value in frame[column].fillna(""):
        counts.update(tokenize(value))

    return counts


# ============================================================
# 6. COUNTRY PROCESSING
# ============================================================

def build_country_mapping():

    mapping = {}

    for country in pycountry.countries:

        names = [country.name]

        for attr in ("official_name", "common_name"):
            if hasattr(country, attr):
                names.append(getattr(country, attr))

        for name in names:
            mapping[name.lower()] = country.alpha_3

    mapping.update(COUNTRY_ALIASES)

    return mapping


COUNTRY_MAPPING = build_country_mapping()


def extract_countries(text):

    text = str(text).lower()
    found = set()

    for name, iso3 in sorted(
        COUNTRY_MAPPING.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):

        pattern = (
            r"(?<![a-z])"
            + re.escape(name)
            + r"(?![a-z])"
        )

        if re.search(pattern, text):
            found.add(iso3)

    return found


def country_name(code):

    try:
        return pycountry.countries.get(
            alpha_3=code
        ).name
    except Exception:
        return code


# Fallback centroids for frequently encountered research countries.
# The countryinfo package supplies the full list; these values keep the map
# usable if that optional dataset cannot be loaded on the hosting platform.
FALLBACK_COUNTRY_COORDINATES = {
    "ARG": (-64.0, -34.0), "AUS": (134.0, -25.0),
    "AUT": (14.0, 47.5), "BEL": (4.7, 50.5),
    "BRA": (-52.0, -10.0), "CAN": (-106.0, 56.0),
    "CHE": (8.2, 46.8), "CHL": (-71.0, -30.0),
    "CHN": (105.0, 35.0), "COL": (-72.0, 4.0),
    "CZE": (15.5, 49.8), "DEU": (10.4, 51.1),
    "DNK": (9.5, 56.0), "EGY": (30.0, 26.0),
    "ESP": (-4.0, 40.0), "FIN": (26.0, 64.0),
    "FRA": (2.0, 46.0), "GBR": (-2.0, 54.0),
    "GRC": (22.0, 39.0), "HUN": (19.0, 47.0),
    "IDN": (118.0, -2.0), "IND": (79.0, 22.0),
    "IRL": (-8.0, 53.0), "ISR": (35.0, 31.0),
    "ITA": (12.5, 42.5), "JPN": (138.0, 37.0),
    "KOR": (127.8, 36.0), "MEX": (-102.0, 23.0),
    "MYS": (102.0, 4.0), "NLD": (5.5, 52.2),
    "NOR": (10.0, 62.0), "NZL": (172.0, -41.0),
    "POL": (19.0, 52.0), "PRT": (-8.0, 39.5),
    "RUS": (100.0, 60.0), "SAU": (45.0, 24.0),
    "SGP": (103.8, 1.35), "SWE": (15.0, 62.0),
    "TUR": (35.0, 39.0), "TWN": (121.0, 24.0),
    "UKR": (32.0, 49.0), "USA": (-98.0, 39.5),
    "ZAF": (24.0, -29.0),
}


@st.cache_data(show_spinner=False)
def get_country_coordinates():
    """Return local ISO-3 centroids as {code: {Latitude, Longitude}}.

    This intentionally avoids CountryInfo: its API differs between package
    versions and can require a country_name argument during initialization.
    The local dictionary makes deployment deterministic and works offline.
    """
    return {
        code: {"Latitude": float(lat), "Longitude": float(lon)}
        for code, (lon, lat) in FALLBACK_COUNTRY_COORDINATES.items()
    }


def great_circle_points(lon1, lat1, lon2, lat2, n=36):
    """Calculate points along a great-circle route, splitting at the dateline."""

    lon1_r, lat1_r, lon2_r, lat2_r = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    p1 = np.array([
        np.cos(lat1_r) * np.cos(lon1_r),
        np.cos(lat1_r) * np.sin(lon1_r),
        np.sin(lat1_r),
    ])
    p2 = np.array([
        np.cos(lat2_r) * np.cos(lon2_r),
        np.cos(lat2_r) * np.sin(lon2_r),
        np.sin(lat2_r),
    ])

    dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-10:
        return [lon1, lon2], [lat1, lat2]

    if np.isclose(np.sin(omega), 0):
        # Antipodal or nearly antipodal points: straight interpolation fallback.
        raw_lons = np.linspace(lon1, lon2, n)
        raw_lats = np.linspace(lat1, lat2, n)
    else:
        points = []
        for t in np.linspace(0, 1, n):
            point = (
                np.sin((1 - t) * omega) / np.sin(omega) * p1
                + np.sin(t * omega) / np.sin(omega) * p2
            )
            point /= np.linalg.norm(point)
            points.append(point)

        points = np.asarray(points)
        raw_lons = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        raw_lats = np.degrees(
            np.arctan2(
                points[:, 2],
                np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2),
            )
        )

    lons, lats = [], []
    previous_lon = None

    for lon, lat in zip(raw_lons, raw_lats):
        lon = float(lon)
        lat = float(lat)

        # Break the line where it crosses the ±180° meridian.
        if previous_lon is not None and abs(lon - previous_lon) > 180:
            lons.append(None)
            lats.append(None)

        lons.append(lon)
        lats.append(lat)
        previous_lon = lon

    return lons, lats


# ============================================================
# 7. PLOTLY DESIGN HELPERS
# ============================================================

def chart_layout(fig, height=400, legend=True):

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=15, r=20, t=35, b=20),
        font=dict(
            family="DM Sans, sans-serif",
            color="#344054",
            size=12
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#172033",
            font_size=12,
            font_color="white"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title_text=""
        ) if legend else dict(visible=False)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EDF1F6",
        zeroline=False,
        title=None
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title=None
    )

    return fig


def section(title, subtitle=None):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-sub">{subtitle}</div>',
            unsafe_allow_html=True
        )


def kpi(label, value, note=""):

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def horizontal_bar(data, category, value, height=400):

    if data.empty:
        return None

    fig = px.bar(
        data.sort_values(value),
        x=value,
        y=category,
        orientation="h",
        text=value,
        color_discrete_sequence=COLORS
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False
    )

    return chart_layout(
        fig,
        height=height,
        legend=False
    )


def show_chart(fig, key=None):

    if fig is not None:

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "scrollZoom": False,
                "responsive": True,
            }
        )


# ============================================================
# 8. FILE UPLOAD
# ============================================================

with st.sidebar:

    st.markdown("## 📂 Dataset")

    uploaded_file = st.file_uploader(
        "Import Scopus CSV",
        type=["csv"]
    )

    st.caption(
        "Compatible with Scopus exports including "
        "Abstract and keyword fields."
    )

if uploaded_file is None:

    st.info(
        "Importe ton fichier CSV Scopus pour construire "
        "ton tableau de bord bibliométrique."
    )

    st.markdown("""
    **Champs pris en charge :**

    - Title, Year, Source title
    - Authors, Author full names, Author(s) ID
    - Affiliations, Authors with affiliations
    - Abstract
    - Author Keywords
    - Index Keywords
    - Cited by, DOI
    - Publisher, Document Type, Open Access
    """)

    st.stop()


try:

    library, original_columns = prepare_data(
        uploaded_file.getvalue()
    )

except Exception as error:

    st.error(f"CSV import error: {error}")
    st.stop()


# Journal database enrichment
journal_db = load_journal_database()

library["Journal_key"] = (
    library["Journal"]
    .str.strip()
    .str.lower()
)

if not journal_db.empty:

    library = library.merge(
        journal_db,
        on="Journal_key",
        how="left",
        suffixes=("", "_database")
    )

    if "Publisher_database" in library.columns:
        library["Publisher"] = (
            library["Publisher"]
            .replace("", np.nan)
            .fillna(library["Publisher_database"])
            .fillna("Unknown")
        )

    if "Domain" not in library.columns:
        library["Domain"] = ""

else:

    if "Publisher" not in library.columns:
        library["Publisher"] = "Unknown"

    library["Domain"] = ""


# ============================================================
# 9. DYNAMIC FILTERS
# ============================================================

with st.sidebar:

    st.divider()
    st.markdown("## 🎛️ Filters")

    years = sorted(
        library["Year"].dropna().astype(int).unique()
    )

    filtered = library.copy()

    if years:

        year_range = st.slider(
            "Publication period",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )

        filtered = filtered[
            filtered["Year"].between(
                year_range[0],
                year_range[1]
            )
            | filtered["Year"].isna()
        ]

    journal_options = sorted(
        filtered["Journal"].dropna().unique()
    )

    selected_journals = st.multiselect(
        "Journals",
        journal_options
    )

    if selected_journals:
        filtered = filtered[
            filtered["Journal"].isin(selected_journals)
        ]

    type_options = sorted(
        filtered["Document Type"].dropna().unique()
    )

    selected_types = st.multiselect(
        "Document types",
        type_options
    )

    if selected_types:
        filtered = filtered[
            filtered["Document Type"].isin(selected_types)
        ]

    publisher_options = sorted(
        x for x in filtered["Publisher"].dropna().unique()
        if x and x != "Unknown"
    )

    selected_publishers = st.multiselect(
        "Publishers",
        publisher_options
    )

    if selected_publishers:
        filtered = filtered[
            filtered["Publisher"].isin(selected_publishers)
        ]

    st.divider()

    st.caption(
        f"{len(filtered)} publications selected "
        f"/ {len(library)} total"
    )


if filtered.empty:

    st.warning(
        "Aucune publication ne correspond aux filtres sélectionnés."
    )

    st.stop()


# ============================================================
# 10. GLOBAL METRICS
# ============================================================

total_pubs = len(filtered)

# Ensure citation values are numeric before calculating bibliometric indicators.
citation_values = (
    pd.to_numeric(filtered["Cited by"], errors="coerce")
    .fillna(0)
    .clip(lower=0)
    .astype(int)
)

total_citations = int(citation_values.sum())

average_citations = (
    total_citations / total_pubs
    if total_pubs else 0
)

# h-index: the largest h for which at least h publications have >= h citations.
sorted_citations = np.sort(citation_values.to_numpy())[::-1]
h_index = int(
    np.sum(sorted_citations >= np.arange(1, len(sorted_citations) + 1))
)

# i10-index: number of publications with at least 10 citations.
i10_index = int((citation_values >= 10).sum())

total_journals = filtered["Journal"].nunique()

authors = {
    author
    for value in filtered["Authors"]
    for author in split_scopus(value)
}

total_authors = len(authors)

valid_years = filtered["Year"].dropna()

period = (
    f"{int(valid_years.min())}–{int(valid_years.max())}"
    if not valid_years.empty else "N/A"
)

abstract_count = int(
    filtered["Abstract"].str.len().gt(0).sum()
)

keyword_count = int(
    (
        filtered["Author Keywords"].str.len().gt(0)
        | filtered["Index Keywords"].str.len().gt(0)
    ).sum()
)

# Precompute author counts
author_counts = Counter()

for value in filtered["Authors"]:
    author_counts.update(split_scopus(value))


# ============================================================
# 11. DASHBOARD TABS
# ============================================================

overview, topics, collaborations, impact, records = st.tabs([
    "📊 Overview",
    "🧬 Research themes",
    "🌍 Collaborations",
    "📚 Sources & impact",
    "🗂️ Publications"
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with overview:

    section(
        "Publication overview",
        "A snapshot of your scientific publication portfolio."
    )

    cols = st.columns(5)

    metrics = [
        ("Publications", f"{total_pubs:,}", period),
        ("Citations", f"{total_citations:,}",
         f"{average_citations:.1f} per publication"),
        ("Authors", f"{total_authors:,}", "Unique author names"),
        ("Journals", f"{total_journals:,}", "Distinct sources"),
        ("Abstracts", f"{abstract_count / total_pubs:.0%}",
         f"{abstract_count} records available")
    ]

    for col, (label, value, note) in zip(cols, metrics):

        with col:
            kpi(label, value, note)

    st.write("")

    left, right = st.columns([1.4, 1])

    # Publication trajectory
    with left:

        section(
            "Publication trajectory",
            "Annual publication output."
        )

        trend = (
            filtered.dropna(subset=["Year"])
            .groupby("Year")
            .size()
            .reset_index(name="Publications")
            .sort_values("Year")
        )

        if not trend.empty:

            trend["Year"] = trend["Year"].astype(int)

            fig = px.area(
                trend,
                x="Year",
                y="Publications",
                markers=True
            )

            fig.update_traces(
                line=dict(color="#536DFE", width=3),
                fillcolor="rgba(83,109,254,0.13)",
                marker=dict(size=8)
            )

            show_chart(
                chart_layout(fig, 360, False),
                "overview_trend"
            )

    # Document types
    with right:

        section(
            "Document types",
            "Distribution of Scopus document categories."
        )

        type_data = (
            filtered["Document Type"]
            .value_counts()
            .reset_index()
        )

        type_data.columns = [
            "Document type", "Publications"
        ]

        if not type_data.empty:

            fig = px.pie(
                type_data,
                names="Document type",
                values="Publications",
                hole=0.65,
                color_discrete_sequence=COLORS
            )

            fig.update_traces(
                textposition="outside",
                textinfo="percent+label",
                marker=dict(
                    line=dict(color="white", width=3)
                )
            )

            fig.add_annotation(
                text=f"<b>{total_pubs}</b><br>papers",
                showarrow=False,
                font=dict(size=20, color="#172033")
            )

            show_chart(
                chart_layout(fig, 360),
                "overview_types"
            )

    st.divider()

    left, right = st.columns(2)

    # Journals
    with left:

        section("Most represented journals")

        data = (
            filtered["Journal"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        data.columns = ["Journal", "Publications"]

        show_chart(
            horizontal_bar(
                data, "Journal", "Publications", 400
            ),
            "overview_journals"
        )

    # Authors
    with right:

        section("Most active authors")

        data = pd.DataFrame(
            author_counts.most_common(10),
            columns=["Author", "Publications"]
        )

        show_chart(
            horizontal_bar(
                data, "Author", "Publications", 400
            ),
            "overview_authors"
        )


# ============================================================
# TAB 2 — RESEARCH THEMES
# ============================================================

with topics:

    section(
        "Research themes",
        "Explore scientific topics using abstracts and Scopus keywords."
    )

    author_keywords = keyword_counter(
        filtered, "Author Keywords"
    )

    indexed_keywords = keyword_counter(
        filtered, "Index Keywords"
    )

    abstract_words = text_counter(
        filtered, "Abstract"
    )

    title_words = text_counter(
        filtered, "Title"
    )

    cols = st.columns(3)

    with cols[0]:
        kpi(
            "Records with keywords",
            f"{keyword_count}/{total_pubs}",
            "Author + indexed keywords"
        )

    with cols[1]:
        kpi(
            "Records with abstract",
            f"{abstract_count}/{total_pubs}",
            "Abstract availability"
        )

    with cols[2]:
        kpi(
            "Distinct author keywords",
            f"{len(author_keywords):,}",
            "Unique keyword expressions"
        )

    st.write("")

    theme_source = st.radio(
        "Thematic analysis source",
        [
            "Author keywords",
            "Indexed keywords",
            "Abstract vocabulary",
            "Title vocabulary"
        ],
        horizontal=True
    )

    sources = {
        "Author keywords": author_keywords,
        "Indexed keywords": indexed_keywords,
        "Abstract vocabulary": abstract_words,
        "Title vocabulary": title_words
    }

    counts = sources[theme_source]

    if counts:

        n_terms = st.slider(
            "Number of terms",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )

        term_data = pd.DataFrame(
            counts.most_common(n_terms),
            columns=["Term", "Frequency"]
        )

        left, right = st.columns([1.2, 1])

        with left:

            section("Most frequent terms")

            show_chart(
                horizontal_bar(
                    term_data,
                    "Term",
                    "Frequency",
                    max(420, n_terms * 24)
                ),
                "themes_terms"
            )

        with right:

            section("Thematic composition")

            fig = px.treemap(
                term_data.head(15),
                path=["Term"],
                values="Frequency",
                color="Frequency",
                color_continuous_scale="Blues"
            )

            show_chart(
                chart_layout(fig, 420, False),
                "themes_treemap"
            )

    else:

        st.info(
            f"No terms available for {theme_source}."
        )

    # Keyword evolution
    st.divider()

    section(
        "Keyword evolution over time",
        "Track the evolution of author keywords across publication years."
    )

    available_keywords = sorted(author_keywords.keys())

    if available_keywords and filtered["Year"].notna().any():

        default_keywords = [
            keyword
            for keyword, _ in author_keywords.most_common(5)
        ]

        selected_keywords = st.multiselect(
            "Select keywords to follow",
            available_keywords,
            default=default_keywords[:5],
            max_selections=8
        )

        if selected_keywords:

            rows = []

            for _, row in filtered.dropna(
                subset=["Year"]
            ).iterrows():

                year = int(row["Year"])

                paper_keywords = {
                    x.lower()
                    for x in split_scopus(
                        row["Author Keywords"]
                    )
                }

                for keyword in selected_keywords:

                    if keyword in paper_keywords:

                        rows.append({
                            "Year": year,
                            "Keyword": keyword
                        })

            evolution = pd.DataFrame(rows)

            if not evolution.empty:

                evolution = (
                    evolution.groupby(["Year", "Keyword"])
                    .size()
                    .reset_index(name="Publications")
                )

                fig = px.line(
                    evolution,
                    x="Year",
                    y="Publications",
                    color="Keyword",
                    markers=True,
                    color_discrete_sequence=COLORS
                )

                fig.update_traces(
                    line=dict(width=2.5),
                    marker=dict(size=7)
                )

                show_chart(
                    chart_layout(fig, 420),
                    "themes_evolution"
                )

            else:

                st.info(
                    "No selected keyword appears in the filtered records."
                )

    else:

        st.info(
            "Keyword evolution requires Author Keywords and publication years."
        )


# ============================================================
# TAB 3 — INTERNATIONAL COLLABORATIONS
# ============================================================

with collaborations:

    section(
        "International collaborations",
        "Geographical distribution and co-publication networks."
    )

    country_counts = Counter()
    paper_countries = []

    for index, affiliation in filtered["Affiliations"].items():

        found = extract_countries(affiliation)

        country_counts.update(found)

        paper_countries.append((index, found))

    if country_counts:

        # --------------------------------------------------------
        # COUNTRY METRICS
        # --------------------------------------------------------

        country_data = pd.DataFrame(
            country_counts.items(),
            columns=["ISO-3", "Publications"]
        )

        country_data["ISO-3"] = (
            country_data["ISO-3"].astype(str).str.strip().str.upper()
        )
        country_data["Country"] = country_data["ISO-3"].map(country_name)

        # Citations are attributed to every country represented in a paper.
        country_citations = Counter()
        for paper_index, countries_in_paper in paper_countries:
            paper_row = filtered.loc[paper_index]
            if isinstance(paper_row, pd.DataFrame):
                citations = int(paper_row["Cited by"].fillna(0).sum())
            else:
                citations = int(paper_row["Cited by"] or 0)
            for iso3 in countries_in_paper:
                country_citations[iso3] += citations

        country_data["Citations"] = (
            country_data["ISO-3"].map(country_citations).fillna(0).astype(int)
        )

        # Build country collaboration pairs from papers containing >= 2 countries.
        pairs = Counter()
        for _, countries_in_paper in paper_countries:
            for country_a, country_b in itertools.combinations(
                sorted(countries_in_paper), 2
            ):
                pairs[(country_a, country_b)] += 1

        pair_data = pd.DataFrame(
            [
                (a, b, country_name(a), country_name(b), count)
                for (a, b), count in pairs.items()
            ],
            columns=[
                "ISO A", "ISO B", "Country 1", "Country 2",
                "Shared publications"
            ]
        )

        if not pair_data.empty:
            pair_data = pair_data.sort_values(
                "Shared publications", ascending=False
            ).reset_index(drop=True)

        # --------------------------------------------------------
        # GLOBAL RESEARCH FOOTPRINT — INTERNATIONAL COLLABORATIONS
        # --------------------------------------------------------

        st.divider()
        section(
            "Global Research Footprint",
            "Scientific output and international co-publication links across countries."
        )

        coordinates = get_country_coordinates()

        country_data["Latitude"] = country_data["ISO-3"].map(
            lambda code: coordinates.get(code, {}).get("Latitude")
        )
        country_data["Longitude"] = country_data["ISO-3"].map(
            lambda code: coordinates.get(code, {}).get("Longitude")
        )

        map_country_data = country_data.dropna(
            subset=["Latitude", "Longitude"]
        ).copy()

        if map_country_data.empty:
            st.warning(
                "No coordinates match the ISO-3 country codes found in this dataset. "
                "Check the country aliases and the country codes extracted from Affiliations."
            )
        else:
            control1, control2 = st.columns(2)

            with control1:
                max_connections = st.slider(
                    "Maximum connections displayed",
                    min_value=10,
                    max_value=150,
                    value=60,
                    step=10,
                    key="map_max_connections"
                )

            with control2:
                maximum_shared = (
                    int(pair_data["Shared publications"].max())
                    if not pair_data.empty else 1
                )
                min_collaborations = st.slider(
                    "Minimum shared publications",
                    min_value=1,
                    max_value=max(1, maximum_shared),
                    value=1,
                    key="map_min_collaborations"
                )

            fig = go.Figure()

            # Select the strongest links after applying the threshold.
            if not pair_data.empty:
                visible_pairs = pair_data[
                    pair_data["Shared publications"] >= min_collaborations
                ].head(max_connections)
            else:
                visible_pairs = pair_data

            if not visible_pairs.empty:
                max_weight = max(
                    1, int(visible_pairs["Shared publications"].max())
                )

                for _, connection in visible_pairs.iterrows():
                    coord_a = coordinates.get(connection["ISO A"])
                    coord_b = coordinates.get(connection["ISO B"])

                    if not coord_a or not coord_b:
                        continue

                    lon_values, lat_values = great_circle_points(
                        coord_a["Longitude"], coord_a["Latitude"],
                        coord_b["Longitude"], coord_b["Latitude"]
                    )

                    weight = int(connection["Shared publications"])
                    strength = weight / max_weight
                    line_width = 0.7 + 2.3 * strength

                    # Subtle halo, then the visible collaboration arc.
                    fig.add_trace(go.Scattergeo(
                        lon=lon_values,
                        lat=lat_values,
                        mode="lines",
                        line=dict(
                            width=line_width + 2.5,
                            color="rgba(14,116,144,0.18)"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    ))

                    fig.add_trace(go.Scattergeo(
                        lon=lon_values,
                        lat=lat_values,
                        mode="lines",
                        line=dict(
                            width=line_width,
                            color=f"rgba(15,118,140,{0.42 + 0.48 * strength:.3f})"
                        ),
                        hovertemplate=(
                            f"<b>{connection['Country 1']}</b> ↔ "
                            f"<b>{connection['Country 2']}</b>"
                            f"<br>Shared publications: {weight}"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))

            # Outer halo around each country node.
            publication_sizes = 8 + 5 * np.log1p(
                map_country_data["Publications"].astype(float)
            )

            fig.add_trace(go.Scattergeo(
                lon=map_country_data["Longitude"],
                lat=map_country_data["Latitude"],
                mode="markers",
                marker=dict(
                    size=publication_sizes + 10,
                    color="#287F9B",
                    opacity=0.13,
                    line=dict(width=0)
                ),
                hoverinfo="skip",
                showlegend=False
            ))

            # Main nodes: size = publications, colour = citations.
            fig.add_trace(go.Scattergeo(
                lon=map_country_data["Longitude"],
                lat=map_country_data["Latitude"],
                mode="markers",
                marker=dict(
                    size=publication_sizes,
                    sizemode="diameter",
                    color=map_country_data["Citations"],
                    colorscale=[
                        [0.00, "#B8CBDD"],
                        [0.30, "#6F9DBB"],
                        [0.65, "#287F9B"],
                        [1.00, "#344E86"]
                    ],
                    opacity=0.96,
                    line=dict(width=1.2, color="#FFFFFF"),
                    colorbar=dict(
                        title=dict(text="Citations", font=dict(color="#334155")),
                        tickfont=dict(color="#475569"),
                        thickness=12,
                        len=0.58,
                        bgcolor="rgba(255,255,255,0.88)",
                        outlinewidth=0
                    )
                ),
                text=map_country_data["Country"],
                customdata=np.column_stack([
                    map_country_data["Publications"],
                    map_country_data["Citations"]
                ]),
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<br>Publications: %{customdata[0]:,.0f}"
                    "<br>Citations: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

            fig.update_geos(
                projection_type="natural earth",
                showframe=False,
                showcoastlines=False,
                showcountries=True,
                countrycolor="#F4F7FA",
                countrywidth=0.9,
                showland=True,
                landcolor="#AFC0D0",
                showocean=True,
                oceancolor="#DCE6EF",
                showlakes=True,
                lakecolor="#D2E0EA",
                bgcolor="#DCE6EF"
            )

            fig.update_layout(
                height=620,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="#DCE6EF",
                plot_bgcolor="#DCE6EF",
                font=dict(family="DM Sans, sans-serif", color="#26384A"),
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#6F9DBB",
                    font=dict(color="#26384A", size=12)
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "displaylogo": False,
                    "scrollZoom": False,
                    "responsive": True
                },
                key="global_research_footprint"
            )

        # --------------------------------------------------------
        # COUNTRY AND COLLABORATION RANKINGS
        # --------------------------------------------------------

        left, right = st.columns(2)

        with left:
            section("Top contributing countries")

            top_countries = (
                country_data.sort_values("Publications", ascending=False)
                .head(15)
                .sort_values("Publications")
            )

            show_chart(
                horizontal_bar(
                    top_countries, "Country", "Publications", 470
                ),
                "collaboration_countries"
            )

        with right:
            section("Strongest international collaborations")

            if not pair_data.empty:
                top_pairs = pair_data.head(15).copy()
                top_pairs["Collaboration"] = (
                    top_pairs["Country 1"] + " – " + top_pairs["Country 2"]
                )
                top_pairs = top_pairs.sort_values("Shared publications")

                show_chart(
                    horizontal_bar(
                        top_pairs, "Collaboration", "Shared publications", 470
                    ),
                    "collaboration_pairs"
                )
            else:
                st.info("No international co-publications detected.")

        # --------------------------------------------------------
        # DETAILED COLLABORATION TABLE
        # --------------------------------------------------------

        st.divider()
        section(
            "International co-publications",
            "Countries appearing together in the same publication."
        )

        if not pair_data.empty:
            st.dataframe(
                pair_data[[
                    "Country 1", "Country 2", "Shared publications"
                ]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No international co-publication detected.")

    else:

        st.info(
            "No country could be identified in the Affiliations field."
        )

    # Co-authorship network
    st.divider()

    section(
        "Co-authorship network",
        "Authors are connected when they share at least one publication."
    )

    if NETWORKX_AVAILABLE:

        graph = nx.Graph()

        for value in filtered["Authors"]:

            paper_authors = list(
                dict.fromkeys(split_scopus(value))
            )

            for author in paper_authors:
                graph.add_node(author)

            for a, b in itertools.combinations(
                paper_authors, 2
            ):

                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1

                else:
                    graph.add_edge(a, b, weight=1)

        if graph.number_of_edges():

            max_authors = st.slider(
                "Maximum number of authors in the network",
                min_value=10,
                max_value=60,
                value=30,
                step=5
            )

            selected_nodes = [
                author
                for author, degree in sorted(
                    graph.degree,
                    key=lambda x: x[1],
                    reverse=True
                )[:max_authors]
            ]

            subgraph = graph.subgraph(
                selected_nodes
            ).copy()

            positions = nx.spring_layout(
                subgraph,
                seed=17,
                k=0.8,
                iterations=80
            )

            edge_x, edge_y = [], []

            for a, b in subgraph.edges():

                x0, y0 = positions[a]
                x1, y1 = positions[b]

                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=1,
                    color="#CBD5E1"
                ),
                hoverinfo="none"
            )

            node_x, node_y = [], []
            node_text, node_sizes = [], []

            for node in subgraph.nodes():

                x, y = positions[node]

                node_x.append(x)
                node_y.append(y)

                node_text.append(
                    f"{node}<br>"
                    f"Publications: {author_counts[node]}<br>"
                    f"Co-authors: {subgraph.degree[node]}"
                )

                node_sizes.append(
                    12 + 4 * author_counts[node]
                )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=list(subgraph.nodes()),
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                marker=dict(
                    size=node_sizes,
                    color=[
                        author_counts[n]
                        for n in subgraph.nodes()
                    ],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Papers"),
                    line=dict(
                        width=1,
                        color="white"
                    )
                ),
                textfont=dict(
                    size=9,
                    color="#344054"
                )
            )

            fig = go.Figure(
                [edge_trace, node_trace]
            )

            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

            fig.update_layout(
                showlegend=False,
                hovermode="closest"
            )

            show_chart(
                chart_layout(fig, 650, False),
                "coauthor_network"
            )

        else:

            st.info(
                "Not enough co-authorship data to build a network."
            )

    else:

        st.warning(
            "Install networkx to activate the co-authorship network: "
            "pip install networkx"
        )


# ============================================================
# TAB 4 — SOURCES AND IMPACT
# ============================================================

with impact:

    section(
        "Sources & impact",
        "Journal distribution, publishers and citation impact."
    )

    left, right = st.columns(2)

    with left:

        section("Publishers")

        publisher_data = (
            filtered["Publisher"]
            .replace("", "Unknown")
            .value_counts()
            .head(12)
            .reset_index()
        )

        publisher_data.columns = [
            "Publisher", "Publications"
        ]

        show_chart(
            horizontal_bar(
                publisher_data,
                "Publisher",
                "Publications",
                420
            ),
            "impact_publishers"
        )

    with right:

        section(
            "Citation indicators",
            "Bibliometric indicators calculated from the publications currently selected."
        )

        indicator_cols = st.columns(3)

        with indicator_cols[0]:
            kpi(
                "Total citations",
                f"{total_citations:,}",
                "Sum of citations"
            )

        with indicator_cols[1]:
            kpi(
                "h-index",
                f"{h_index:,}",
                "Publications cited at least h times"
            )

        with indicator_cols[2]:
            kpi(
                "i10-index",
                f"{i10_index:,}",
                "Publications with ≥ 10 citations"
            )

    st.divider()

    left, right = st.columns(2)

    with left:

        section("Citation distribution")

        citation_data = filtered.copy()

        citation_data["Citation group"] = pd.cut(
            citation_data["Cited by"],
            bins=[-1, 0, 5, 10, 25, 50, np.inf],
            labels=[
                "0", "1–5", "6–10",
                "11–25", "26–50", ">50"
            ]
        )

        citation_groups = (
            citation_data["Citation group"]
            .value_counts(sort=False)
            .reset_index()
        )

        citation_groups.columns = [
            "Citations", "Publications"
        ]

        fig = px.bar(
            citation_groups,
            x="Citations",
            y="Publications",
            text="Publications",
            color_discrete_sequence=["#00A6A6"]
        )

        fig.update_traces(
            textposition="outside"
        )

        show_chart(
            chart_layout(fig, 360, False),
            "impact_citations"
        )

    with right:

        section("Most cited publications")

        top = (
            filtered
            .sort_values("Cited by", ascending=False)
            .head(8)
            .copy()
        )

        top["Short title"] = (
            top["Title"].str.slice(0, 55)
        )

        show_chart(
            horizontal_bar(
                top,
                "Short title",
                "Cited by",
                360
            ),
            "impact_top_papers"
        )

    st.divider()

    section("Journal and publisher summary")

    journal_summary = (
        filtered.groupby(
            ["Journal", "Publisher"],
            dropna=False
        )
        .agg(
            Publications=("Title", "size"),
            Citations=("Cited by", "sum"),
            Mean_citations=("Cited by", "mean")
        )
        .reset_index()
        .sort_values(
            ["Publications", "Citations"],
            ascending=False
        )
    )

    journal_summary["Mean_citations"] = (
        journal_summary["Mean_citations"].round(1)
    )

    st.dataframe(
        journal_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5 — PUBLICATIONS AND EXPORT
# ============================================================

with records:

    section(
        "Publication records",
        "Search and inspect your filtered Scopus publications."
    )

    search = st.text_input(
        "Search title, authors, journal, abstract or keywords"
    )

    display = filtered.copy()

    if search.strip():

        searchable_columns = [
            "Title",
            "Authors",
            "Journal",
            "Abstract",
            "Author Keywords",
            "Index Keywords"
        ]

        mask = display[searchable_columns].apply(
            lambda col: col.str.contains(
                re.escape(search),
                case=False,
                na=False
            )
        ).any(axis=1)

        display = display[mask]

    display_columns = [
        "Year",
        "Title",
        "Authors",
        "Journal",
        "Cited by",
        "Author Keywords",
        "DOI URL"
    ]

    st.dataframe(
        display[display_columns].sort_values(
            ["Year", "Cited by"],
            ascending=[False, False]
        ),
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "DOI URL": st.column_config.LinkColumn(
                "DOI",
                display_text="Open DOI"
            ),
            "Title": st.column_config.TextColumn(
                "Title",
                width="large"
            ),
            "Authors": st.column_config.TextColumn(
                "Authors",
                width="large"
            ),
            "Author Keywords": st.column_config.TextColumn(
                "Author keywords",
                width="large"
            )
        }
    )

    st.divider()

    section("Export")

    export_columns = [
        col for col in filtered.columns
        if col != "Text for topics"
    ]

    csv_data = filtered[
        export_columns
    ].to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download filtered dataset (CSV)",
        data=csv_data,
        file_name="whats_up_science_filtered.csv",
        mime="text/csv"
    )

    st.caption(
        f"{len(original_columns)} original Scopus columns detected."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "What's up Science? · Scopus bibliometric dashboard"
)
