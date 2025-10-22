import streamlit as st
import pandas as pd
from typing import Optional


def get_active_data() -> Optional[pd.DataFrame]:
    """Return the active dataset from session_state in priority order.
    Checks `processed_data`, then `uploaded_data`, then `data`.
    """
    data = st.session_state.get("processed_data")
    if data is None:
        data = st.session_state.get("uploaded_data")
    if data is None:
        data = st.session_state.get("data")
    return data