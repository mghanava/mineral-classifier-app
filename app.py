"""Streamlit entry point for the mineral classifier."""

import streamlit as st

from dashboard.state import CONFIGURE_PAGE, RESULTS_PAGE, RUN_PAGE


def app() -> None:
    """Register the primary destinations and render the selected page."""
    st.set_page_config(page_title="Mineral Classifier", layout="wide")
    page = st.navigation(
        [
            st.Page("dashboard/views/home.py", title="Home", default=True),
            st.Page(CONFIGURE_PAGE, title="Configure"),
            st.Page(RUN_PAGE, title="Run"),
            st.Page(RESULTS_PAGE, title="Results"),
        ],
        position="top",
    )
    # Address native navigation focus and caption contrast with two scoped overrides.
    st.html("""
        <style>
        header a[data-testid="stTopNavLink"]:focus-visible {
            outline: 2px solid currentColor;
            outline-offset: 2px;
        }
        [data-testid="stCaptionContainer"] {
            opacity: .84;
        }
        </style>
    """)
    with (
        st.container(horizontal=True, horizontal_alignment="center"),
        st.container(width=1260, gap="small"),
    ):
        page.run()


if __name__ == "__main__":
    app()
