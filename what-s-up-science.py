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
