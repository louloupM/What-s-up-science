
import re
from io import BytesIO
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pycountry

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="What's up Science?",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

JOURNAL_URL = (
    "https://raw.githubusercontent.com/louloupM/"
    "what-s-up-science/main/"
    "Scientific%20journal%20list%201.2.csv"
)

CHART_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False
}

CHART_LAYOUT = {
    "font": {
        "family": "Arial, sans-serif",
        "size": 12,
        "color": "#344054"
    },
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": dict(l=15, r=15, t=35, b=15),
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 12
    }
}

COLORS = [
    "#3976A8",
    "#5B9BD5",
    "#70AD47",
    "#ED7D31",
    "#A5A5A5",
    "#8064A2",
    "#4BACC6",
    "#C0504D"
]

STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "also", "however", "therefore", "thus",
    "respectively", "significantly", "results",
    "result", "conclusion", "conclusions",
    "purpose", "methodology", "objective",
    "objectives", "present", "reported",
    "study", "studies", "article", "paper",
    "research", "using", "used", "based",
    "showed", "show", "demonstrated",
    "obtained", "performed", "investigated",
    "proposed", "described", "compared",
    "increase", "increased", "decrease",
    "different", "various", "several"
})

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
    "bolivia": "BOL",
    "tanzania": "TZA",
    "syria": "SYR",
    "venezuela": "VEN",
    "moldova": "MDA",
    "czech republic": "CZE"
}


# ============================================================
# CSS / INTERFACE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 750;
    color: #1D2939;
    margin-bottom: 2px;
}

.subtitle {
    text-align: center;
    color: #667085;
    font-size: 15px;
    margin-bottom: 30px;
}

section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
}

div[data-testid="stMetric"] {
    background-color: #F8FAFC;
    border: 1px solid #EAECF0;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stMetric"] label {
    color: #667085;
}

h2, h3 {
    color: #344054;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🔬 What\'s up Science?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Scientific publication analytics dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CSV IMPORT
# ============================================================

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes):

    errors = []

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):

        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding,
                sep=None,
                engine="python"
            )

        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        "Unable to read CSV: " + " | ".join(errors)
    )


def find_column(df, candidates):

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


@st.cache_data(show_spinner=False)
def prepare_data(file_bytes):

    raw = read_uploaded_csv(file_bytes)

    raw.columns = [
        str(col).strip()
        for col in raw.columns
    ]

    column_map = {

        "Title": find_column(
            raw, ["Title"]
        ),

        "Year": find_column(
            raw, ["Year", "Publication Year"]
        ),

        "Journal": find_column(
            raw, ["Source title", "Journal"]
        ),

        "Cited by": find_column(
            raw, ["Cited by", "Citations", "Times Cited"]
        ),

        "DOI": find_column(
            raw, ["DOI"]
        ),

        "Affiliations": find_column(
            raw, ["Affiliations", "Author Affiliations"]
        ),

        "Authors": find_column(
            raw, ["Authors", "Author(s)"]
        ),

        "Author full names": find_column(
            raw, ["Author full names", "Authors full names"]
        ),

        "Author(s) ID": find_column(
            raw, ["Author(s) ID", "Author ID"]
        ),

        "Authors with affiliations": find_column(
            raw, ["Authors with affiliations"]
        ),

        "Document Type": find_column(
            raw, ["Document Type", "Type"]
        ),

        "Abstract": find_column(
            raw, ["Abstract"]
        ),

        "Author Keywords": find_column(
            raw, ["Author Keywords", "Authors Keywords"]
        ),

        "Indexed Keywords": find_column(
            raw, ["Index Keywords", "Indexed Keywords"]
        ),

        "Publisher": find_column(
            raw, ["Publisher"]
        ),

        "Open Access": find_column(
            raw, ["Open Access"]
        ),

        "EID": find_column(
            raw, ["EID"]
        ),

        "Link": find_column(
            raw, ["Link"]
        ),

        "Language": find_column(
            raw, ["Language of Original Document", "Language"]
        )
    }

    required = ["Title", "Year", "Journal"]

    missing = [
        name for name in required
        if column_map[name] is None
    ]

    if missing:

        raise ValueError(
            "Missing required Scopus columns: "
            + ", ".join(missing)
        )

    df = pd.DataFrame(index=raw.index)

    for target, source in column_map.items():

        if source is not None:
            df[target] = raw[source]

        else:
            df[target] = np.nan

    # Preserve all original Scopus fields
    for col in raw.columns:

        if col not in df.columns:
            df[col] = raw[col]

    # Cleaning
    text_columns = [
        "Title",
        "Journal",
        "Affiliations",
        "Authors",
        "Author full names",
        "Abstract",
        "Author Keywords",
        "Indexed Keywords",
        "Authors with affiliations",
        "Document Type",
        "DOI"
    ]

    for col in text_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df["Cited by"] = pd.to_numeric(
        df["Cited by"],
        errors="coerce"
    ).fillna(0)

    df["Journal"] = df["Journal"].replace(
        "", "Unknown journal"
    )

    df["Document Type"] = df["Document Type"].replace(
        "", "Unknown"
    )

    df["DOI URL"] = df["DOI"].apply(
        lambda x: (
            x if x.startswith("http")
            else "https://doi.org/" + x if x
            else ""
        )
    )

    return df


# ============================================================
# JOURNAL DATABASE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_journal_database():

    try:

        journals = pd.read_csv(JOURNAL_URL)

        journals.columns = journals.columns.str.strip()

        if (
            len(journals.columns) < 5
            or "Title" not in journals.columns
        ):
            return pd.DataFrame()

        publisher_col = journals.columns[3]
        domain_col = journals.columns[4]

        result = journals[
            ["Title", publisher_col, domain_col]
        ].copy()

        result.columns = [
            "Journal",
            "Publisher_database",
            "Domain"
        ]

        result["Journal_key"] = (
            result["Journal"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return result[
            ["Journal_key", "Publisher_database", "Domain"]
        ].drop_duplicates("Journal_key")

    except Exception:

        return pd.DataFrame()


# ============================================================
# COUNTRY PROCESSING
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


def extract_countries(affiliation_text):

    text = str(affiliation_text).lower()

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


def split_authors(value):

    if not value:
        return []

    return [
        author.strip()
        for author in str(value).split(";")
        if author.strip()
    ]


# ============================================================
# SCIENTIFIC CONCEPT EXTRACTION
# ============================================================

def extract_scientific_concepts(
    abstracts,
    author_keywords=None,
    indexed_keywords=None,
    min_df=2,
    max_features=300
):

    documents = []

    for i, abstract in enumerate(abstracts):

        parts = []

        if pd.notna(abstract):
            parts.append(str(abstract))

        if author_keywords is not None:

            if pd.notna(author_keywords.iloc[i]):
                parts.append(str(author_keywords.iloc[i]))

        if indexed_keywords is not None:

            if pd.notna(indexed_keywords.iloc[i]):
                parts.append(str(indexed_keywords.iloc[i]))

        documents.append(" ".join(parts))

    documents = [
        doc.lower()
        for doc in documents
        if doc.strip()
    ]

    if len(documents) < 2:

        return pd.DataFrame(
            columns=["Concept", "Frequency", "Score"]
        )

    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 4),
        min_df=min_df,
        max_df=0.90,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        sublinear_tf=True
    )

    try:

        matrix = vectorizer.fit_transform(documents)

    except ValueError:

        return pd.DataFrame(
            columns=["Concept", "Frequency", "Score"]
        )

    terms = vectorizer.get_feature_names_out()

    frequencies = (matrix > 0).sum(axis=0).A1

    scores = matrix.mean(axis=0).A1

    concepts = pd.DataFrame({

        "Concept": terms,

        "Frequency": frequencies,

        "Score": scores

    })

    concepts = concepts.sort_values(
        ["Frequency", "Score"],
        ascending=False
    )

    concepts = concepts[
        concepts["Frequency"] >= min_df
    ]

    return concepts.reset_index(drop=True)


# ============================================================
# CHART HELPER
# ============================================================

def display_chart(fig, height=None):

    fig.update_layout(**CHART_LAYOUT)

    if height is not None:
        fig.update_layout(height=height)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=CHART_CONFIG
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your Scopus CSV export",
    type=["csv"],
    help=(
        "Scopus CSV export with Title, Year and Source title. "
        "Abstract and keyword fields are recommended."
    )
)

if uploaded_file is None:

    st.info(
        "Import your Scopus CSV file to start the analysis."
    )

    st.markdown("""
    **Supported Scopus fields**

    - Title, Year, Source title
    - Cited by, DOI
    - Authors, Author full names
    - Affiliations, Authors with affiliations
    - Abstract
    - Author Keywords
    - Indexed Keywords
    - Document Type
    - Publisher
    """)

    st.stop()


try:

    library = prepare_data(
        uploaded_file.getvalue()
    )

except Exception as error:

    st.error(f"Error importing file: {error}")
    st.stop()


# ============================================================
# ENRICHMENT
# ============================================================

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
        how="left"
    )

else:

    library["Publisher_database"] = np.nan
    library["Domain"] = np.nan

# Prefer Scopus publisher if available
if "Publisher" in library.columns:

    library["Publisher_final"] = (
        library["Publisher"]
        .replace("", np.nan)
        .fillna(library["Publisher_database"])
    )

else:

    library["Publisher_final"] = library["Publisher_database"]


# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.header("Filters")

years = sorted(
    library["Year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

if years:

    selected_years = st.sidebar.slider(
        "Publication years",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years))
    )

    filtered = library[
        library["Year"].between(
            selected_years[0],
            selected_years[1]
        )
        | library["Year"].isna()
    ].copy()

else:

    filtered = library.copy()


journals = sorted(
    filtered["Journal"].dropna().unique().tolist()
)

selected_journals = st.sidebar.multiselect(
    "Journals",
    options=journals,
    default=[]
)

if selected_journals:

    filtered = filtered[
        filtered["Journal"].isin(selected_journals)
    ]


doc_types = sorted(
    filtered["Document Type"].dropna().unique().tolist()
)

selected_types = st.sidebar.multiselect(
    "Document types",
    options=doc_types,
    default=[]
)

if selected_types:

    filtered = filtered[
        filtered["Document Type"].isin(selected_types)
    ]


publishers = sorted(
    filtered["Publisher_final"]
    .dropna()
    .unique()
    .tolist()
)

selected_publishers = st.sidebar.multiselect(
    "Publishers",
    options=publishers,
    default=[]
)

if selected_publishers:

    filtered = filtered[
        filtered["Publisher_final"].isin(selected_publishers)
    ]


if filtered.empty:

    st.warning(
        "No publications match the selected filters."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.markdown("## Publication overview")

total_publications = len(filtered)

total_citations = int(
    filtered["Cited by"].sum()
)

average_citations = round(
    total_citations / total_publications,
    1
)

total_journals = filtered["Journal"].nunique()

total_authors = len({

    author

    for value in filtered["Authors"]

    for author in split_authors(value)

})

valid_years = filtered["Year"].dropna()

period = (

    f"{int(valid_years.min())} – "
    f"{int(valid_years.max())}"

    if not valid_years.empty
    else "N/A"
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Publications",
    f"{total_publications:,}"
)

c2.metric(
    "Total citations",
    f"{total_citations:,}"
)

c3.metric(
    "Citations / publication",
    average_citations
)

c4.metric(
    "Journals",
    f"{total_journals:,}"
)

c5.metric(
    "Authors",
    f"{total_authors:,}"
)

st.caption(f"Publication period: {period}")

st.divider()


# ============================================================
# SCIENTIFIC CONCEPTS
# ============================================================

st.markdown("## Scientific concepts")

st.caption(
    "Extraction of recurring scientific concepts and expressions "
    "from abstracts and Scopus keywords."
)

concepts_df = extract_scientific_concepts(
    filtered["Abstract"],
    filtered["Author Keywords"],
    filtered["Indexed Keywords"],
    min_df=2,
    max_features=300
)

if not concepts_df.empty:

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("### Most frequent scientific concepts")

        concept_chart = (
            concepts_df
            .head(20)
            .sort_values("Frequency")
        )

        fig = px.bar(
            concept_chart,
            x="Frequency",
            y="Concept",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Blues",
            labels={
                "Frequency": "Publications",
                "Concept": "Scientific concept"
            }
        )

        fig.update_layout(
            coloraxis_showscale=False,
            yaxis_title="",
            xaxis_title="Number of publications"
        )

        display_chart(fig, height=650)

    with col2:

        st.markdown("### Concept ranking")

        st.dataframe(
            concepts_df.head(30),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Concept": st.column_config.TextColumn(
                    "Scientific concept",
                    width="large"
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Publications",
                    format="%d"
                ),
                "Score": st.column_config.NumberColumn(
                    "TF-IDF score",
                    format="%.4f"
                )
            }
        )

else:

    st.info(
        "Not enough recurring concepts found. "
        "Check that your Scopus export includes abstracts "
        "or keywords."
    )


st.divider()


# ============================================================
# KEYWORD ANALYSIS
# ============================================================

st.markdown("## Author and indexed keywords")

keyword_counter = Counter()

for value in filtered["Author Keywords"]:

    if pd.notna(value):

        keywords = re.split(r";|\|", str(value))

        keyword_counter.update({

            keyword.strip().lower()

            for keyword in keywords

            if keyword.strip()

        })


for value in filtered["Indexed Keywords"]:

    if pd.notna(value):

        keywords = re.split(r";|\|", str(value))

        keyword_counter.update({

            keyword.strip().lower()

            for keyword in keywords

            if keyword.strip()

        })


if keyword_counter:

    keyword_df = pd.DataFrame(
        keyword_counter.most_common(25),
        columns=["Keyword", "Frequency"]
    )

    fig = px.bar(
        keyword_df.sort_values("Frequency"),
        x="Frequency",
        y="Keyword",
        orientation="h",
        color="Frequency",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Frequency",
        yaxis_title=""
    )

    display_chart(fig, height=650)

else:

    st.info("No author or indexed keywords found.")


st.divider()


# ============================================================
# PUBLICATION TRENDS
# ============================================================

st.markdown("## Publication analysis")

left, right = st.columns(2)

with left:

    st.markdown("### Publications by year")

    year_df = (
        filtered.dropna(subset=["Year"])
        .groupby("Year")
        .size()
        .reset_index(name="Publications")
        .sort_values("Year")
    )

    if not year_df.empty:

        year_df["Year"] = (
            year_df["Year"]
            .astype(int)
            .astype(str)
        )

        fig = px.bar(
            year_df,
            x="Year",
            y="Publications",
            text="Publications",
            color_discrete_sequence=[COLORS[1]]
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Publications",
            showlegend=False
        )

        display_chart(fig, height=400)


with right:

    st.markdown("### Top 15 journals")

    journal_df = (
        filtered.groupby("Journal")
        .size()
        .reset_index(name="Publications")
        .sort_values(
            "Publications",
            ascending=False
        )
        .head(15)
        .sort_values("Publications")
    )

    if not journal_df.empty:

        fig = px.bar(
            journal_df,
            x="Publications",
            y="Journal",
            orientation="h",
            color_discrete_sequence=[COLORS[1]]
        )

        fig.update_layout(
            xaxis_title="Publications",
            yaxis_title=""
        )

        display_chart(fig, height=550)


st.divider()


# ============================================================
# PUBLISHERS AND SCIENTIFIC DOMAINS
# ============================================================

st.markdown("## Publishers and scientific domains")

left, right = st.columns(2)

with left:

    st.markdown("### Publishers")

    publisher_df = (
        filtered.dropna(subset=["Publisher_final"])
        .groupby("Publisher_final")
        .size()
        .reset_index(name="Publications")
        .sort_values(
            "Publications",
            ascending=False
        )
        .head(12)
    )

    if not publisher_df.empty:

        fig = px.pie(
            publisher_df,
            names="Publisher_final",
            values="Publications",
            hole=0.45
        )

        display_chart(fig, height=450)

    else:

        st.info("Publisher information unavailable.")


with right:

    st.markdown("### Scientific domains")

    domain_df = (
        filtered.dropna(subset=["Domain"])
        .groupby("Domain")
        .size()
        .reset_index(name="Publications")
        .sort_values(
            "Publications",
            ascending=False
        )
    )

    if not domain_df.empty:

        fig = px.pie(
            domain_df,
            names="Domain",
            values="Publications",
            hole=0.45
        )

        display_chart(fig, height=450)

    else:

        st.info(
            "Scientific domain information unavailable. "
            "Check the journal metadata database."
        )


st.divider()


# ============================================================
# INTERNATIONAL COLLABORATIONS
# ============================================================

st.markdown("## International collaborations")

country_counts = Counter()

for affiliations in filtered["Affiliations"]:

    country_counts.update(
        extract_countries(affiliations)
    )


if country_counts:

    world_df = pd.DataFrame(
        country_counts.items(),
        columns=["iso_alpha", "Publications"]
    )

    world_df["Country"] = world_df["iso_alpha"].apply(
        lambda code: (
            pycountry.countries.get(alpha_3=code).name
            if pycountry.countries.get(alpha_3=code)
            else code
        )
    )

    fig = px.choropleth(
        world_df,
        locations="iso_alpha",
        locationmode="ISO-3",
        color="Publications",
        hover_name="Country",
        color_continuous_scale="Blues",
        projection="equirectangular"
    )

    fig.update_geos(
        projection_type="equirectangular",
        showcoastlines=True,
        coastlinecolor="white",
        showland=True,
        landcolor="#F0F2F5",
        showocean=True,
        oceancolor="#FFFFFF",
        showcountries=True,
        countrycolor="white",
        showframe=False,
        showlakes=False
    )

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title="Publications"
        )
    )

    display_chart(fig)

    st.markdown("### Most represented countries")

    country_table = (
        world_df.sort_values(
            "Publications",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        country_table[
            ["Country", "Publications"]
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No country could be identified in the Affiliations column."
    )


st.divider()


# ============================================================
# AUTHOR ANALYSIS
# ============================================================

st.markdown("## Author analysis")

author_counts = Counter()

for authors_text in filtered["Authors"]:

    author_counts.update(
        split_authors(authors_text)
    )


if author_counts:

    author_df = pd.DataFrame(
        author_counts.most_common(20),
        columns=["Author", "Publications"]
    )

    fig = px.bar(
        author_df.sort_values("Publications"),
        x="Publications",
        y="Author",
        orientation="h",
        color_discrete_sequence=[COLORS[1]]
    )

    fig.update_layout(
        xaxis_title="Publications",
        yaxis_title=""
    )

    display_chart(fig, height=600)

else:

    st.info("No author information found.")


st.divider()


# ============================================================
# CO-AUTHORSHIP ANALYSIS
# ============================================================

st.markdown("## Co-authorship analysis")

coauthor_pairs = Counter()

for authors_text in filtered["Authors"]:

    authors = split_authors(authors_text)

    if len(authors) > 1:

        for pair in combinations(
            sorted(set(authors)),
            2
        ):

            coauthor_pairs[pair] += 1


if coauthor_pairs:

    collaboration_df = pd.DataFrame(
        [
            {
                "Author 1": pair[0],
                "Author 2": pair[1],
                "Joint publications": count
            }
            for pair, count in coauthor_pairs.most_common(30)
        ]
    )

    st.markdown("### Most frequent co-author relationships")

    st.dataframe(
        collaboration_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No co-authorship relationships could be extracted."
    )


st.divider()


# ============================================================
# DOCUMENT TYPES
# ============================================================

st.markdown("## Document types")

type_df = (
    filtered.groupby("Document Type")
    .size()
    .reset_index(name="Publications")
    .sort_values(
        "Publications",
        ascending=False
    )
)

if not type_df.empty:

    fig = px.pie(
        type_df,
        names="Document Type",
        values="Publications",
        hole=0.45
    )

    display_chart(fig, height=450)


st.divider()


# ============================================================
# MOST CITED PUBLICATIONS
# ============================================================

st.markdown("## Top 10 most cited publications")

top_publications = (
    filtered.sort_values(
        "Cited by",
        ascending=False
    )
    .head(10)
    [
        [
            "Title",
            "Year",
            "Cited by",
            "DOI URL",
            "Journal",
            "Authors"
        ]
    ]
    .copy()
)

top_publications["Year"] = (
    top_publications["Year"]
    .apply(
        lambda value: (
            int(value)
            if pd.notna(value)
            else None
        )
    )
)

st.dataframe(
    top_publications,
    use_container_width=True,
    hide_index=True,
    column_config={

        "Title": st.column_config.TextColumn(
            "Publication title",
            width="large"
        ),

        "Cited by": st.column_config.NumberColumn(
            "Citations",
            format="%d"
        ),

        "DOI URL": st.column_config.LinkColumn(
            "DOI",
            display_text="Open DOI",
            width="small"
        ),

        "Authors": st.column_config.TextColumn(
            "Authors",
            width="large"
        )
    }
)


st.divider()


# ============================================================
# PUBLICATION TABLE
# ============================================================

st.markdown("## Publication database")

search_text = st.text_input(
    "Search publications",
    placeholder="Search by title, author or journal..."
)

publication_table = filtered.copy()

if search_text:

    mask = (
        publication_table["Title"]
        .str.contains(
            search_text,
            case=False,
            na=False
        )
        |
        publication_table["Authors"]
        .str.contains(
            search_text,
            case=False,
            na=False
        )
        |
        publication_table["Journal"]
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    )

    publication_table = publication_table[mask]


st.dataframe(
    publication_table[
        [
            "Title",
            "Year",
            "Journal",
            "Authors",
            "Cited by",
            "DOI URL"
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Title": st.column_config.TextColumn(
            "Publication title",
            width="large"
        ),
        "DOI URL": st.column_config.LinkColumn(
            "DOI",
            display_text="Open DOI"
        )
    }
)


# ============================================================
# EXPORT
# ============================================================

st.divider()

st.markdown("## Export")

export_df = filtered.drop(
    columns=["Journal_key"],
    errors="ignore"
)

csv_data = export_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "Download analyzed dataset (CSV)",
    data=csv_data,
    file_name="whats_up_science_analysis.csv",
    mime="text/csv"
)

st.caption(
    "What's up Science? — Scopus bibliometric analysis"
)
