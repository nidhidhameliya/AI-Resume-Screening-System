from __future__ import annotations

import streamlit as st


def render_pdf_preview(uploaded_file) -> None:
    try:
        import fitz
    except ImportError:
        st.info("Install PyMuPDF to enable PDF preview support.")
        return

    try:
        document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        st.caption(f"Page count: {len(document)}")
        if len(document) > 0:
            first_page = document.load_page(0)
            pix = first_page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
            st.image(pix.tobytes("png"))
    except Exception as exc:
        st.warning(f"PDF preview unavailable: {exc}")
