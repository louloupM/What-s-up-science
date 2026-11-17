import re
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import pycountry
import streamlit as st
import matplotlib.pyplot as plt
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
    "https://raw.githubusercontent.com/louloupM/what-s-up-science/"
    "main/Scientific%20journal%20list%201.2.csv"
)

STOPWORDS = {
    "the", "and", "for", "with", "from", "using", "based", "into",
    "through", "study", "studies", "effect", "effects", "analysis",
    "characterization", "evaluation", "development", "novel", "new",
    "high", "low", "different", "role", "approach", "method", "methods",
    "production", "application", "applications", "properties", "activity",
    "activities", "investigation", "towards", "via", "between", "under",
    "over", "after", "before", "during", "its", "their", "this", "that",
    "these", "those", "are", "was", "were", "has", "have", "been", "can",
    "may", "not", "one", "two", "three", "on", "in", "of", "to", "a",
    "an", "by", "as", "at", "or", "is"
}

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

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
.main-title {
    text-align: center; font-size: 38px; font-weight: 700;
    color: #20252B; margin-bottom: 4px;
}
.subtitle {
    text-align: center; color: #68737D;
    font-size: 16px; margin-bottom: 28px;
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
# DATA IMPORT AND NORMALIZATION
# Supports Scopus CSV exports and the former generic format.
# ============================================================

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes):
    """Read a CSV while trying common Scopus/export encodings."""
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
    raise ValueError("Impossible de lire le CSV. " + " | ".join(errors))


def find_column(df, candidates):
    """Return the first matching column, ignoring case and spaces."""
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
    raw.columns = [str(c).strip() for c in raw.columns]

    # Scopus uses "Source title"; older input used "Journal".
    column_map = {
        "Title": find_column(raw, ["Title"]),
        "Abstract": find_column(raw, ["Abstract"]),
        "Year": find_column(raw, ["Year", "Publication Year"]),
        "Journal": find_column(raw, ["Source title", "Journal", "Source Title"]),
        "Cited by": find_column(raw, ["Cited by", "Citations", "Times Cited"]),
        "DOI": find_column(raw, ["DOI"]),
        "Affiliations": find_column(raw, ["Affiliations", "Author Affiliations"]),
        "Authors": find_column(raw, ["Authors", "Author(s)"]),
        "Author full names": find_column(raw, ["Author full names", "Authors full names"]),
        "Document Type": find_column(raw, ["Document Type", "Type"]),
        "EID": find_column(raw, ["EID"]),
        "Link": find_column(raw, ["Link"]),
        "Open Access": find_column(raw, ["Open Access"]),
    }

    required = ["Title", "Year", "Journal", "Abstract"]
    missing = [name for name in required if column_map[name] is None]
    if missing:
        raise ValueError(
            "Colonnes indispensables absentes : "
            + ", ".join(missing)
            + ". Le fichier doit notamment contenir Title, Year, Abstract et "
              "Source title (ou Journal)."
        )

    df = pd.DataFrame()
    for target, source in column_map.items():
        if source is not None:
            df[target] = raw[source]
        else:
            df[target] = np.nan

    # Keep all original Scopus fields as well, for the export.
    for col in raw.columns:
        if col not in df.columns:
            df[col] = raw[col]

    df["Title"] = df["Title"].fillna("").astype(str).str.strip()
    df["Abstract"] = df["Abstract"].fillna("").astype(str).str.strip()
    df["Journal"] = (
        df["Journal"].fillna("Unknown journal").astype(str).str.strip()
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0)
    df["Affiliations"] = df["Affiliations"].fillna("").astype(str)
    df["Authors"] = df["Authors"].fillna("").astype(str)
    df["Author full names"] = df["Author full names"].fillna("").astype(str)
    df["DOI"] = df["DOI"].fillna("").astype(str).str.strip()
    df["Document Type"] = df["Document Type"].fillna("Unknown").astype(str)

    # Normalize DOI links for clickable table cells.
    df["DOI URL"] = df["DOI"].apply(
        lambda x: x if x.startswith("http") else (
            "https://doi.org/" + x if x else ""
        )
    )

    return df


# ============================================================
# OPTIONAL JOURNAL DATABASE: publisher and domain enrichment
# These fields are not provided as standard Scopus columns.
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_journal_database():
    try:
        journals = pd.read_csv(JOURNAL_URL)
        journals.columns = journals.columns.str.strip()

        # Preserve the column positions used in the original app:
        # 4th column = publisher, 5th column = scientific domain.
        if len(journals.columns) < 5 or "Title" not in journals.columns:
            return pd.DataFrame(columns=["Journal_key", "Publisher", "Domain"])

        publisher_col = journals.columns[3]
        domain_col = journals.columns[4]

        result = journals[["Title", publisher_col, domain_col]].copy()
        result.columns = ["Journal", "Publisher", "Domain"]
        result["Journal_key"] = (
            result["Journal"].fillna("").astype(str).str.strip().str.lower()
        )
        return result[["Journal_key", "Publisher", "Domain"]].drop_duplicates(
            "Journal_key"
        )
    except Exception:
        return pd.DataFrame(columns=["Journal_key", "Publisher", "Domain"])


# ============================================================
# TEXT AND COUNTRY PROCESSING
# ============================================================

def clean_title(title):
    title = str(title).lower()
    title = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", title)
    title = re.sub(r"[^a-z\s-]", " ", title)
    return [
        word for word in title.split()
        if len(word) > 2 and word not in STOPWORDS
    ]


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
    """Return unique ISO3 country codes found in one publication's affiliations."""
    text = str(affiliation_text).lower()
    found = set()

    # Longest names first, reducing partial-name matching issues.
    for name, iso3 in sorted(
        COUNTRY_MAPPING.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = r"(?<![a-z])" + re.escape(name) + r"(?![a-z])"
        if re.search(pattern, text):
            found.add(iso3)

    return found


def split_scopus_authors(authors_text):
    """Scopus 'Authors' field is generally separated by semicolons."""
    if not authors_text:
        return []
    return [
        author.strip()
        for author in str(authors_text).split(";")
        if author.strip()
    ]


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your Scopus CSV export",
    type=["csv"],
    help="Export from Scopus in CSV format, including at least Title, Year "
         "and Source title."
)

if uploaded_file is None:
    st.info("Importe ton fichier CSV exporté depuis Scopus pour commencer.")
    st.markdown("""
    **Colonnes Scopus prises en charge :**
    - Title, Year, Source title
    - Cited by, DOI, Affiliations
    - Authors, Author full names
    - Document Type, Open Access, EID
    """)
    st.stop()

try:
    library = prepare_data(uploaded_file.getvalue())
except Exception as error:
    st.error(f"Erreur lors de la lecture du fichier : {error}")
    st.stop()

# Enrich with publisher/domain when the optional journal database is reachable.
journal_db = load_journal_database()
library["Journal_key"] = library["Journal"].str.strip().str.lower()
if not journal_db.empty:
    library = library.merge(journal_db, on="Journal_key", how="left")
else:
    library["Publisher"] = np.nan
    library["Domain"] = np.nan


# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.header("Filters")

years = sorted(library["Year"].dropna().astype(int).unique().tolist())
if years:
    selected_years = st.sidebar.slider(
        "Publication years",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    filtered = library[
        library["Year"].between(selected_years[0], selected_years[1])
        | library["Year"].isna()
    ].copy()
else:
    filtered = library.copy()

journals = sorted(filtered["Journal"].dropna().unique().tolist())
selected_journals = st.sidebar.multiselect(
    "Journals",
    options=journals,
    default=[]
)
if selected_journals:
    filtered = filtered[filtered["Journal"].isin(selected_journals)]

doc_types = sorted(filtered["Document Type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Document types",
    options=doc_types,
    default=[]
)
if selected_types:
    filtered = filtered[filtered["Document Type"].isin(selected_types)]

if filtered.empty:
    st.warning("Aucune publication ne correspond aux filtres sélectionnés.")
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.markdown("## Publication overview")

total_publications = len(filtered)
total_citations = int(filtered["Cited by"].sum())
average_citations = round(total_citations / total_publications, 1)
total_journals = filtered["Journal"].nunique()
total_authors = len({
    author
    for value in filtered["Authors"]
    for author in split_scopus_authors(value)
})

valid_years = filtered["Year"].dropna()
period = (
    f"{int(valid_years.min())} – {int(valid_years.max())}"
    if not valid_years.empty else "N/A"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Publications", f"{total_publications:,}")
c2.metric("Total citations", f"{total_citations:,}")
c3.metric("Citations / publication", average_citations)
c4.metric("Journals", f"{total_journals:,}")
c5.metric("Authors", f"{total_authors:,}")

st.caption(f"Publication period: {period}")
st.divider()


# ============================================================
# RESEARCH TOPICS
# ============================================================

st.markdown("## Research topics")

all_words = []
for abstract in filtered["Abstract"]:
    all_words.extend(clean_abstract(abstract))

word_counts = Counter(all_words)
left, right = st.columns([2, 1])

with left:
    if word_counts:
        cloud = WordCloud(
            width=1200,
            height=500,
            background_color="white",
            max_words=70,
            collocations=False,
            colormap="viridis"
        ).generate_from_frequencies(word_counts)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No usable words found in publication abstract.")

with right:
    st.markdown("### Top 20 keywords")
    keyword_df = pd.DataFrame(
        word_counts.most_common(20),
        columns=["Keyword", "Frequency"]
    )
    if not keyword_df.empty:
        fig = px.bar(
            keyword_df.sort_values("Frequency"),
            x="Frequency",
            y="Keyword",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            height=500,
            coloraxis_showscale=False,
            margin=dict(l=0, r=5, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================
# PUBLICATION TRENDS AND JOURNALS
# ============================================================

st.markdown("## Publication analysis")
left, right = st.columns(2)

with left:
    st.markdown("### Publications by year")
    year_df = (
        filtered.dropna(subset=["Year"])
        .groupby("Year").size()
        .reset_index(name="Publications")
        .sort_values("Year")
    )
    if not year_df.empty:
        year_df["Year"] = year_df["Year"].astype(int).astype(str)
        fig = px.bar(
            year_df, x="Year", y="Publications",
            text="Publications",
            color_discrete_sequence=["#65A9D8"]
        )
        fig.update_layout(
            xaxis_title="Year", yaxis_title="Publications",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Top 15 journals")
    journal_df = (
        filtered.groupby("Journal").size()
        .reset_index(name="Publications")
        .sort_values("Publications", ascending=False)
        .head(15).sort_values("Publications")
    )
    fig = px.bar(
        journal_df, x="Publications", y="Journal",
        orientation="h",
        color_discrete_sequence=["#65A9D8"]
    )
    fig.update_layout(
        xaxis_title="Publications", yaxis_title="",
        showlegend=False, height=550
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================
# PUBLISHERS AND DOMAINS
# ============================================================

st.markdown("## Publishers and scientific domains")
left, right = st.columns(2)

with left:
    st.markdown("### Publishers")
    publisher_df = (
        filtered.dropna(subset=["Publisher"])
        .groupby("Publisher").size()
        .reset_index(name="Publications")
        .sort_values("Publications", ascending=False)
        .head(10)
    )
    if not publisher_df.empty:
        fig = px.pie(
            publisher_df, names="Publisher", values="Publications",
            hole=0.35
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Publisher data unavailable. The Scopus export does not provide "
            "a standard Publisher column; journal metadata could not be matched."
        )

with right:
    st.markdown("### Scientific domains")
    domain_df = (
        filtered.dropna(subset=["Domain"])
        .groupby("Domain").size()
        .reset_index(name="Publications")
        .sort_values("Publications", ascending=False)
    )
    if not domain_df.empty:
        fig = px.pie(
            domain_df, names="Domain", values="Publications",
            hole=0.35
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Domain data unavailable. Check the journal metadata database "
            "or add a Domain column to your source data."
        )

st.divider()


# ============================================================
# INTERNATIONAL COLLABORATIONS
# Count each country once per publication.
# ============================================================

st.markdown("## International collaborations")

country_counts = Counter()
for affiliations in filtered["Affiliations"]:
    country_counts.update(extract_countries(affiliations))

if country_counts:
    world_df = pd.DataFrame(
        country_counts.items(),
        columns=["iso_alpha", "Publications"]
    )

    fig = px.choropleth(
        world_df,
        locations="iso_alpha",
        locationmode="ISO-3",
        color="Publications",
        hover_name="iso_alpha",
        color_continuous_scale="Blues",
        projection="natural earth"
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    country_table = world_df.sort_values(
        "Publications", ascending=False
    ).head(15)
    st.markdown("### Most represented countries")
    st.dataframe(country_table, use_container_width=True, hide_index=True)
else:
    st.info("No country could be identified in the Affiliations column.")

st.divider()


# ============================================================
# AUTHORS
# ============================================================

st.markdown("## Author analysis")

author_counts = Counter()
for authors_text in filtered["Authors"]:
    author_counts.update(split_scopus_authors(authors_text))

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
        color_discrete_sequence=["#65A9D8"]
    )
    fig.update_layout(
        height=600,
        xaxis_title="Publications",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No author information found in the Authors column.")

st.divider()


# ============================================================
# DOCUMENT TYPES
# ============================================================

st.markdown("## Document types")

type_df = (
    filtered.groupby("Document Type").size()
    .reset_index(name="Publications")
    .sort_values("Publications", ascending=False)
)

if not type_df.empty:
    fig = px.pie(
        type_df, names="Document Type", values="Publications",
        hole=0.35
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================
# MOST CITED PUBLICATIONS
# ============================================================

st.markdown("## Top 10 most cited publications")

top_publications = (
    filtered.sort_values("Cited by", ascending=False)
    .head(10)
    [["Title", "Year", "Cited by", "DOI URL", "Journal", "Authors"]]
    .copy()
)

top_publications["Year"] = top_publications["Year"].apply(
    lambda value: int(value) if pd.notna(value) else None
)

st.dataframe(
    top_publications,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Title": st.column_config.TextColumn(
            "Publication title", width="large"
        ),
        "Cited by": st.column_config.NumberColumn(
            "Citations", format="%d"
        ),
        "DOI URL": st.column_config.LinkColumn(
            "DOI", display_text="Open DOI", width="small"
        ),
        "Authors": st.column_config.TextColumn(
            "Authors", width="large"
        )
    }
)


# ============================================================
# EXPORT
# ============================================================

st.divider()
st.markdown("## Export")

export_df = filtered.drop(columns=["Journal_key"], errors="ignore")
csv_data = export_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Download analyzed dataset (CSV)",
    data=csv_data,
    file_name="whats_up_science_analysis.csv",
    mime="text/csv"
)

st.caption("What's up Science? — Scopus bibliometric analysis")
