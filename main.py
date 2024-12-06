import streamlit as st
from streamlit_option_menu import option_menu
import sales_projection, truck_location_prediction, home

st.set_page_config(
    page_title="Truck Sales App",
    page_icon="🚚",
    layout="wide",  
    initial_sidebar_state="expanded" 
)


selected = option_menu(
    menu_title=None,
    options = ['Home', 'Sales Projections', 'Truck Location Prediction'],
    icons=['house-fill', 'graph-up', 'geo-alt'],
    default_index = 0,
    orientation = 'horizontal'
)

if selected == 'Home':
    home.app()
elif selected == 'Sales Projections':
    sales_projection.app()
elif selected == 'Truck Location Prediction':
    truck_location_prediction.app()


