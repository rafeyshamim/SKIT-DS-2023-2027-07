import streamlit as st

def render():
    st.title("Disease Analysis Results")
    st.caption("AI-Powered diagnosis and confidence scoring.")

    scan_id = st.session_state.get("scan_id")
    if not scan_id:
        st.warning("No scan uploaded. Please go back to the Upload screen.")
        if st.button("← Back to Upload"):
            st.session_state.stage = "upload"
            st.rerun()
        return

    # Mocking or fetching results
    st.subheader("Analysis Report")
    
    # In a real app, we would load `analyzer.analyze(...)` result here.
    # For now, let's use the mock session state or static data to demonstrate flow.
    result = st.session_state.get("analysis_result", {
        "predicted_class_label": "Benign / Normal",
        "confidence": 0.89,
        "above_threshold": True,
        "probabilities": {
            "Benign / Normal": 0.89,
            "Malignant": 0.11
        }
    })

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", result["predicted_class_label"])
    col2.metric("Confidence Score", f"{result['confidence'] * 100:.2f}%")
    col3.metric("Meets Threshold", "YES" if result["above_threshold"] else "NO")

    st.divider()
    st.write("### Class Probabilities")
    for label, prob in result["probabilities"].items():
        st.progress(prob, text=f"{label}: {prob * 100:.2f}%")

    st.info("Disclaimer: This is for research and educational purposes only and should not be considered a substitute for diagnosis by a qualified healthcare professional.")

    st.divider()
    if st.button("← Back to Viewer"):
        st.session_state.stage = "viewer"
        st.rerun()
