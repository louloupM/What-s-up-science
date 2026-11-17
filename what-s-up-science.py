
import re
import os
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pycountry

from wordcloud import WordCloud


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="What's up Science?",
    page_icon="🔬",
    layout="wide"
)

JOURNAL_URL = (
    "https://raw.githubusercontent.com/louloupM/"
    "what-s-up-science/main/"
    "Scientific%20journal%20list%201.2.csv"
)

STOPWORDS = {
    "the", "and", "for", "with", "from", "using", "based",
    "into", "through", "study", "studies", "effect", "effects",
    "analysis", "characterization", "evaluation", "development",
    "novel", "new", "high", "low", "different", "role",
    "approach", "method", "methods", "production", "application",
    "applications", "properties", "activity", "activities",
    "investigation", "towards", "via", "between", "under",
    "over", "after", "before", "during", "its", "their",
    "this", "that", "these", "those", "are", "was", "were",
    "has", "have", "been", "can", "may", "not", "on", "in",
    "of", "to", "a", "an", "by", "as", "at", "or", "is"
}

# Common alternative country names
COUNTRY_ALIASES = {
    "USA": "United States",
    "U.S.A.": "United States",
    "US": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Russia": "Russian Federation",
    "Iran": "Iran, Islamic Republic of",
    "Vietnam": "Viet Nam",
    "Taiwan": "Taiwan, Province of China",
    "Bolivia": "Bolivia, Plurinational State of",
    "Tanzania": "Tanzania, United Republic of",
    "Syria": "Syrian Arab Republic",
    "Venezuela": "Venezuela, Bolivarian Republic of",
    "Moldova": "Moldova, Republic of"
}


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #20252B;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #68737D;
    font-size: 16px;
    margin-bottom: 30px;
}
.metric-card {
    background-color: #F5F8FA;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🔬 What\'s up Science?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Scientific publication analytics dashboard</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=86400)
def load_journal_database():

    return pd.read_csv(JOURNAL_URL)


@st.cache_data(show_spinner=False)
def prepare_data(file_bytes):

    from io import BytesIO

    df = pd.read_csv(BytesIO(file_bytes))

    # Required columns
    required_columns = ["Title", "Year", "Journal"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    # Optional columns
    for col in ["Cited by", "DOI", "Affiliations"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean data
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Journal"] = df["Journal"].fillna("Unknown").astype(str)

    df["Year"] = pd.to_numeric(
        df["Year"], errors="coerce"
    )

    df["Cited by"] = pd.to_numeric(
        df["Cited by"], errors="coerce"
    ).fillna(0)

    df["Affiliations"] = (
        df["Affiliations"].fillna("").astype(str)
    )

    df["DOI"] = df["DOI"].fillna("").astype(str)

    return df


# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_title(title):

    title = title.lower()

    # Remove text in brackets
    title = re.sub(r"\([^)]*\)", " ", title)
    title = re.sub(r"\[[^]]*\]", " ", title)

    # Remove punctuation and numbers
    title = re.sub(r"[^a-z\s-]", " ", title)

    words = title.split()

    words = [
        word for word in words
        if len(word) > 2
        and word not in STOPWORDS
    ]

    return words


# ============================================================
# COUNTRY EXTRACTION
# ============================================================

def get_country_mapping():

    mapping = {}

    for country in pycountry.countries:

        names = [
            country.name
        ]

        if hasattr(country, "official_name"):
            names.append(country.official_name)

        if hasattr(country, "common_name"):
            names.append(country.common_name)

        for name in names:
            mapping[name.lower()] = country.alpha_3

    for alias, official_name in COUNTRY_ALIASES.items():

        try:
            country = pycountry.countries.lookup(official_name)
            mapping[alias.lower()] = country.alpha_3

        except LookupError:
            pass

    return mapping


COUNTRY_MAPPING = get_country_mapping()


def extract_countries(affiliation):

    text = affiliation.lower()

    found = set()

    for name, iso_code in COUNTRY_MAPPING.items():

        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"

        if re.search(pattern, text):
            found.add(iso_code)

    return found


# ============================================================
# JOURNAL DATABASE
# ============================================================

@st.cache_data(ttl=86400)
def prepare_journal_database():

    journals = load_journal_database()

    journals.columns = journals.columns.str.strip()

    if "Title" not in journals.columns:
        raise ValueError(
            "The journal database does not contain a Title column."
        )

    # Preserve the original database structure:
    # Column 4 = Publisher
    # Column 5 = Domain

    if len(journals.columns) < 5:
        raise ValueError(
            "The journal database must contain at least 5 columns."
        )

    publisher_column = journals.columns[3]
    domain_column = journals.columns[4]

    journals = journals[
        ["Title", publisher_column, domain_column]
    ].copy()

    journals.columns = ["Journal", "Publisher", "Domain"]

    journals["Journal"] = (
        journals["Journal"].fillna("").astype(str).str.strip().str.lower()
    )

    journals = journals.drop_duplicates(
        subset="Journal"
    )

    return journals


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your bibliographic dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload a CSV file containing your scientific publications."
    )

    st.markdown("""
    **Expected columns:**
    - Title
    - Year
    - Journal
    - Cited by
    - DOI
    - Affiliations
    """)

    st.stop()


try:

    library = prepare_data(uploaded_file.getvalue())

    journal_database = prepare_journal_database()

except Exception as error:

    st.error(f"Unable to process the dataset: {error}")
    st.stop()


# ============================================================
# GENERAL STATISTICS
# ============================================================

total_publications = len(library)

total_citations = int(library["Cited by"].sum())

average_citations = (
    round(total_citations / total_publications, 1)
    if total_publications else 0
)

total_journals = library["Journal"].nunique()

valid_years = library["Year"].dropna()

if len(valid_years):
    first_year = int(valid_years.min())
    last_year = int(valid_years.max())
else:
    first_year = "N/A"
    last_year = "N/A"


st.markdown("## Publication overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Publications", f"{total_publications:,}")
col2.metric("Total citations", f"{total_citations:,}")
col3.metric("Average citations", average_citations)
col4.metric("Scientific journals", f"{total_journals:,}")
col5.metric("Publication period", f"{first_year} – {last_year}")


st.divider()


# ============================================================
# WORD CLOUD
# ============================================================

st.markdown("## Research topics")

all_words = []

for title in library["Title"]:
    all_words.extend(clean_title(title))

word_counts = Counter(all_words)

col1, col2 = st.columns([2, 1])

with col1:

    if word_counts:

        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color="white",
            max_words=60,
            colormap="viridis",
            collocations=False
        ).generate_from_frequencies(word_counts)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No usable words found in publication titles.")


with col2:

    st.markdown("### Top 20 keywords")

    keywords_df = pd.DataFrame(
        word_counts.most_common(20),
        columns=["Keyword", "Frequency"]
    )

    if not keywords_df.empty:

        fig = px.bar(
            keywords_df.sort_values("Frequency"),
            x="Frequency",
            y="Keyword",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=0, r=10, t=10, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)


st.divider()


# ============================================================
# PUBLICATION ANALYSIS
# ============================================================

st.markdown("## Publication analysis")

col1, col2 = st.columns(2)


# Publications by year

with col1:

    st.markdown("### Publications by year")

    year_df = (
        library.dropna(subset=["Year"])
        .groupby("Year")
        .size()
        .reset_index(name="Publications")
        .sort_values("Year")
    )

    if not year_df.empty:

        year_df["Year"] = year_df["Year"].astype(int).astype(str)

        fig = px.bar(
            year_df,
            x="Year",
            y="Publications",
            text="Publications",
            color_discrete_sequence=["#65A9D8"]
        )

        fig.update_layout(
            xaxis_title="Publication year",
            yaxis_title="Number of publications",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


# Publications by journal

with col2:

    st.markdown("### Top 20 scientific journals")

    journal_df = (
        library.groupby("Journal")
        .size()
        .reset_index(name="Publications")
        .sort_values("Publications", ascending=False)
        .head(20)
        .sort_values("Publications")
    )

    fig = px.bar(
        journal_df,
        x="Publications",
        y="Journal",
        orientation="h",
        color_discrete_sequence=["#65A9D8"]
    )

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Number of publications",
        showlegend=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


st.divider()


# ============================================================
# JOURNAL INFORMATION
# ============================================================

st.markdown("## Scientific publishers and domains")

# Merge journal metadata with publication data

library_enriched = library.copy()

library_enriched["Journal_key"] = (
    library_enriched["Journal"].str.strip().str.lower()
)

journal_database["Journal_key"] = journal_database["Journal"]

library_enriched = library_enriched.merge(
    journal_database[
        ["Journal_key", "Publisher", "Domain"]
    ],
    on="Journal_key",
    how="left"
)


col1, col2 = st.columns(2)


# Publishers

with col1:

    st.markdown("### Top 10 publishers")

    publisher_df = (
        library_enriched.dropna(subset=["Publisher"])
        .groupby("Publisher")
        .size()
        .reset_index(name="Publications")
        .sort_values("Publications", ascending=False)
        .head(10)
    )

    if not publisher_df.empty:

        fig = px.pie(
            publisher_df,
            names="Publisher",
            values="Publications",
            hole=0.35
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No publisher information found.")


# Scientific domains

with col2:

    st.markdown("### Scientific domains")

    domain_df = (
        library_enriched.dropna(subset=["Domain"])
        .groupby("Domain")
        .size()
        .reset_index(name="Publications")
        .sort_values("Publications", ascending=False)
    )

    if not domain_df.empty:

        fig = px.pie(
            domain_df,
            names="Domain",
            values="Publications",
            hole=0.35
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No domain information found.")


st.divider()


# ============================================================
# WORLD MAP
# ============================================================

st.markdown("## International scientific collaborations")

country_counts = Counter()

for affiliation in library["Affiliations"]:

    countries = extract_countries(affiliation)

    # Count one publication per country, even if several
    # affiliations from that country are present.

    country_counts.update(countries)


if country_counts:

    world_df = pd.DataFrame(
        country_counts.items(),
        columns=["iso_alpha", "Publications"]
    )

    fig = px.choropleth(
        world_df,
        locations="iso_alpha",
        color="Publications",
        hover_name="iso_alpha",
        color_continuous_scale="Blues",
        projection="natural earth"
    )

    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=20, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

else:

    st.info(
        "No countries could be identified in the Affiliations column."
    )


st.divider()


# ============================================================
# MOST CITED PUBLICATIONS
# ============================================================

st.markdown("## Top 5 most cited publications")

top5 = (
    library.sort_values(
        "Cited by",
        ascending=False
    )
    .head(5)
    [["Title", "Year", "Cited by", "DOI", "Journal"]]
    .copy()
)

top5["Year"] = top5["Year"].apply(
    lambda x: int(x) if pd.notna(x) else None
)

st.dataframe(
    top5,
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
        "DOI": st.column_config.LinkColumn(
            "DOI",
            display_text="Open DOI",
            width="medium"
        )
    }
)


# ============================================================
# DOWNLOAD ENRICHED DATASET
# ============================================================

st.divider()

st.markdown("## Export data")

csv_data = library_enriched.drop(
    columns=["Journal_key"],
    errors="ignore"
).to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download enriched dataset",
    data=csv_data,
    file_name="scientific_publications_analysis.csv",
    mime="text/csv"
)

st.caption(
    "What's up Science? – Scientific publication analytics"
)
