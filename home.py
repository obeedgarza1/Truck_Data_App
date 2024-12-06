import streamlit as st

def app():
    col1, col2, col3 = st.columns([1,4,1])
    col2.markdown("<h1 style='text-align: center; color: #4A90E2;'>🌮Welcome to the Food Truck Sales App 🚐</h1>", unsafe_allow_html=True)
    
    col2.markdown("""
        <h3 style='text-align: center; color: #333;'>Track sales, find trends and optimal locations, and fuel your food truck success!.</h3>
        """, unsafe_allow_html=True)

    col2.markdown("<hr style='border-top: 3px solid #bbb;'>", unsafe_allow_html=True)
    
    col2.subheader("🍟 About This App")
    col2.write(
        """
        Truck Analysis App is designed to help franchise owners to monitor sales data of Tasty Bites, forecast trends, and identify 
        potential new locations for their trucks. Our goal is to make data-driven decisions easier and 
        more accessible for everyone in the industry.
        """
    )
    
    col2.subheader("💡 Key Features")
    col2.markdown("""
        - **Sales Projections**: Monitor daily sales trends and make informed decisions on stock and supply.
        - **Location Prediction**: Identify hotspots with high demand and plan your food truck locations accordingly.
        - **Custom Filters**: Dive deeper into data with filters by food type, country, and year.
        """)
    
    col2.subheader("🚀 Get Started")
    col2.write(
        """
        Use the navigation menu to explore **Sales Projections** and **Location Prediction** features.
        Uncover insights that can drive your food truck business to new locations and increase profitability.
        """
    )

    col2.markdown(
        """
        <br><br>
        <hr>
        <p style='text-align: center; color: grey;'>
        Check out the source code on <a href='https://github.com/your-username/streamlit-app' target='_blank'>GitHub</a>.
        </p>
        """,
        unsafe_allow_html=True
    )