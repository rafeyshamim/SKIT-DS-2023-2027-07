import os
import streamlit as st
from PIL import Image

def render():
    st.title("3D Image Viewer")
    st.caption("Visualizations generated from your CT Scan.")

    scan_id = st.session_state.get("scan_id")
    if not scan_id:
        st.warning("No scan uploaded. Please go back to the Upload screen.")
        if st.button("← Back to Upload"):
            st.session_state.stage = "upload"
            st.rerun()
        return

    st.success(f"Viewing visualizations for Scan ID: {scan_id}")

    # For now, we will just show placeholders or load actual saved visualizations if available
    recon_dir = "results/reconstruction/"
    if os.path.exists(recon_dir):
        files = [f for f in os.listdir(recon_dir) if f.endswith(".png")]
        if files:
            st.write("### Extracted Slices & MIP Projections")
            cols = st.columns(3)
            for idx, file in enumerate(files[:3]):  # Just show top 3 for UI
                with cols[idx % 3]:
                    st.image(Image.open(os.path.join(recon_dir, file)), caption=file, use_container_width=True)
            
            # HTML 3D model
            html_files = [f for f in os.listdir(recon_dir) if f.endswith(".html")]
            if html_files:
                st.write("### Interactive 3D Isosurface")
                with open(os.path.join(recon_dir, html_files[0]), 'r') as f:
                    html_data = f.read()
                st.components.v1.html(html_data, height=500)
        else:
            st.info("Reconstruction images are being generated or not found.")
    else:
        st.info("No reconstruction results folder found.")

    st.divider()
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("← Upload another scan"):
            st.session_state.stage = "upload"
            st.rerun()
    with cols[1]:
        if st.button("View Analysis Results →", type="primary"):
            st.session_state.stage = "results"
            st.rerun()
