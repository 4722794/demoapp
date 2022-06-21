import streamlit as st

def changefont():
    st.markdown(
            """
        <style>
        @import url('http://fonts.cdnfonts.com/css/pac-font');

        p  {
        font-family: 'PacFont';
        font-size: 15px;
        }
        </style>

        """,
            unsafe_allow_html=True,
        )