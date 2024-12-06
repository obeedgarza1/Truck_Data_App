import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, when
from pyspark.sql import functions as F
import pandas as pd
from pyspark.sql import SparkSession
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
from folium.plugins import HeatMap
import random
import duckdb
import gdown

spark = SparkSession.builder \
    .appName("Truck_Data") \
    .getOrCreate()

# URL of your dataset
url = "https://drive.google.com/uc?id=1_D6T_1hw_G4koH7YzLawoP_J6nn6zHEJ"
source_file = "truck_data.parquet"

# Download the dataset
gdown.download(url, source_file, quiet=False)

selected_columns = ['LATITUDE', 'LONGITUDE', 'PRICE', 'MENU_TYPE', 'ITEM_CATEGORY', 'CITY',
                    'ORDER_TS', 'UNIT_PRICE', 'MENU_ITEM_NAME', 'COST_OF_GOODS_USD', 'QUANTITY']

def load_sample_data(source_file, selected_columns):
    """
    Loads a sample from the dataset to extract unique values for optimization.

    Returns:
        tuple: A tuple containing lists of unique menu types, item categories, and cities.
    """

    df_spark_sample = spark.read.parquet(source_file).select(selected_columns).sample(fraction=0.1, seed=42)
    
    def get_unique_values(df, column_name):
        unique_values = [row[column_name] for row in df.select(column_name).distinct().collect()]
        return unique_values

    menu_types = get_unique_values(df_spark_sample, 'MENU_TYPE')
    item_category = get_unique_values(df_spark_sample, 'ITEM_CATEGORY')
    cities = get_unique_values(df_spark_sample, 'CITY')

    item_category = ["All"] + item_category

    return menu_types, item_category, cities

@st.cache_resource
def load_data(source_file, selected_columns, menu_type, city, selected_category):
    """
    Loads and filters data based on user input, optimizing file reading.

    Filters by menu type, city, and selected item categories, and adds time-related columns.

    Returns:
        DataFrame: A filtered Spark DataFrame with the selected data.
    """

    df_spark = spark.read.parquet(source_file).select(selected_columns)
    
    df_spark = df_spark.withColumn('HOUR', hour(col('ORDER_TS'))) \
                        .withColumn('TIME_OF_DAY', 
                                    when((col('HOUR') >= 5) & (col('HOUR') < 12), 'Morning')
                                    .when((col('HOUR') >= 12) & (col('HOUR') < 17), 'Afternoon')
                                    .when((col('HOUR') >= 17) & (col('HOUR') < 21), 'Evening')
                                    .otherwise('Night')
                                    ) \
                        .withColumn('MONTH', month(col('ORDER_TS'))) \
                        .withColumn('SEASON',
                                    when((col('MONTH').isin(12, 1, 2)), 'Winter')
                                    .when((col('MONTH').isin(3, 4, 5)), 'Spring')
                                    .when((col('MONTH').isin(6, 7, 8)), 'Summer')
                                    .when((col('MONTH').isin(9, 10, 11)), 'Autumn')
                                    ) \
                        .withColumn("DATE", col("ORDER_TS").cast("date")) \
                        .drop('ORDER_TS', 'HOUR', 'MONTH')

    df_spark = df_spark.filter(F.col('MENU_TYPE') == menu_type)
    df_spark = df_spark.filter(F.col('CITY') == city)


    if "All" not in selected_category:  
        df_spark = df_spark.filter(F.col('ITEM_CATEGORY').isin(selected_category))  
    
    return df_spark

def app():
    st.title('Discover Ideal Food Truck Hotspots in Poland')

    st.write(
        """
        Explore prime locations for food truck success across Poland with data-driven insights. This map highlights hotspots by sales performance, customer demand, and menu preferences. Filter by menu type, like tacos or other popular items, to identify optimal areas to set up a food truck and maximize sales potential.
        Use the interactive map and filters to dive deeper into specific regions and menu types, helping you make informed decisions on where to expand or launch your food truck business.
        """
    )

    st.divider()
    menu_types, item_category, cities = load_sample_data(source_file, selected_columns)

    l, ml, mr, r, rr = st.columns([1.0, 1.3, 1.0, 1.0, 1.8])

    selected_menu = l.selectbox(label='Select a Menu Type for your Food Truck', options = menu_types, index = 0)
    selected_category =ml.pills(label='Select the options for your Food Truck', options = item_category, default = ["All"])
    selected_city = mr.segmented_control(label='Select a City for your Food Truck', options = cities, default = 'Warsaw')

    df_spark = load_data(source_file, selected_columns, selected_menu, selected_city, selected_category)

    @st.cache_data
    def get_city_centers():
        """
        Caches the geographical coordinates of major cities.

        Returns:
            dict: A dictionary of cities with their latitude and longitude.
        """
        return {
            'Warsaw': [52.2298, 21.0118], 
            'Krakow': [50.0647, 19.9450] 
        }

    city_centers = get_city_centers()
    city_center = city_centers.get(selected_city, [52.2298, 21.0118])

    data = None

    with r:
        st.write('')
        if r.button('Apply Selection'):
            data = df_spark

    rr.markdown("#### Top 3 Items with Seasonal Sales Insights")

    if data is not None:
        if data.count() > 0:
            grouped_data = data.groupBy('LATITUDE', 'LONGITUDE', 'MENU_TYPE', 'ITEM_CATEGORY', 'SEASON', 'TIME_OF_DAY') \
                .agg(F.sum('PRICE').alias('TOTAL_SALES')) \
                .orderBy(F.desc('LATITUDE'), F.desc('LONGITUDE'), F.asc('MENU_TYPE'))

            df_pandas = grouped_data.toPandas()

            if 'All' in selected_category:
                all_categories = df_pandas['ITEM_CATEGORY'].unique()
            else:
                all_categories = selected_category

            new_locations = []
            for _ in range(17):
                lat_offset = random.uniform(-0.01, 0.01)
                lon_offset = random.uniform(-0.01, 0.01)
                new_lat = city_center[0] + lat_offset
                new_lon = city_center[1] + lon_offset

                season = random.choice(df_pandas['SEASON'].unique())
                time_of_day = random.choice(df_pandas['TIME_OF_DAY'].unique())
                new_sales = random.uniform(df_pandas['TOTAL_SALES'].min(), df_pandas['TOTAL_SALES'].max())

                for category in all_categories:
                    new_locations.append({
                        'LATITUDE': new_lat,
                        'LONGITUDE': new_lon,
                        'MENU_TYPE': selected_menu,
                        'ITEM_CATEGORY': category,
                        'SEASON': season,
                        'TIME_OF_DAY': time_of_day,
                        'TOTAL_SALES': new_sales
                    })
            
            for _ in range(17):
                lat_offset = random.uniform(-0.03, 0.03)
                lon_offset = random.uniform(-0.03, 0.03)
                new_lat = city_center[0] + lat_offset
                new_lon = city_center[1] + lon_offset

                season = random.choice(df_pandas['SEASON'].unique())
                time_of_day = random.choice(df_pandas['TIME_OF_DAY'].unique())
                new_sales = random.uniform(df_pandas['TOTAL_SALES'].min(), df_pandas['TOTAL_SALES'].max())

                for category in all_categories:
                    new_locations.append({
                        'LATITUDE': new_lat,
                        'LONGITUDE': new_lon,
                        'MENU_TYPE': selected_menu,
                        'ITEM_CATEGORY': category,  
                        'SEASON': season,
                        'TIME_OF_DAY': time_of_day,
                        'TOTAL_SALES': new_sales
                    })
            
            for _ in range(17):
                lat_offset = random.uniform(-0.06, 0.06)
                lon_offset = random.uniform(-0.06, 0.06)
                new_lat = city_center[0] + lat_offset
                new_lon = city_center[1] + lon_offset

                season = random.choice(df_pandas['SEASON'].unique())
                time_of_day = random.choice(df_pandas['TIME_OF_DAY'].unique())
                new_sales = random.uniform(df_pandas['TOTAL_SALES'].min(), df_pandas['TOTAL_SALES'].max())

                for category in all_categories:
                    new_locations.append({
                        'LATITUDE': new_lat,
                        'LONGITUDE': new_lon,
                        'MENU_TYPE': selected_menu,
                        'ITEM_CATEGORY': category, 
                        'SEASON': season,
                        'TIME_OF_DAY': time_of_day,
                        'TOTAL_SALES': new_sales
                    })

            new_locations_df = pd.DataFrame(new_locations)

            season_order = {'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4}
            time_order = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}

            new_locations_df['SEASON_ENCODED'] = new_locations_df['SEASON'].map(season_order)
            new_locations_df['TIME_ENCODED'] = new_locations_df['TIME_OF_DAY'].map(time_order)

            df_pandas['SEASON_ENCODED'] = df_pandas['SEASON'].map(season_order)
            df_pandas['TIME_ENCODED'] = df_pandas['TIME_OF_DAY'].map(time_order)

            new_locations_df.drop(columns = ['SEASON', 'TIME_OF_DAY'], inplace = True)
            df_pandas.drop(columns = ['SEASON', 'TIME_OF_DAY'], inplace = True)

            df_pandas = pd.get_dummies(df_pandas, columns = ['MENU_TYPE', 'ITEM_CATEGORY'], drop_first = True)
            new_locations_df = pd.get_dummies(new_locations_df, columns = ['MENU_TYPE', 'ITEM_CATEGORY'], drop_first = True)

            new_locations_df = new_locations_df.reindex(columns=df_pandas.columns, fill_value=0)

            X = df_pandas.drop(columns='TOTAL_SALES')
            y = df_pandas['TOTAL_SALES']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

            model = GradientBoostingRegressor(n_estimators = 100, random_state = 17)
            model.fit(X_train, y_train)

            X_new = new_locations_df.drop(columns = ['TOTAL_SALES'], errors = 'ignore')
            new_locations_df['PREDICTED_SALES'] = model.predict(X_new)

            hotspot_map = folium.Map(location = city_center, zoom_start = 12)

            heatmap_data = new_locations_df[['LATITUDE', 'LONGITUDE', 'PREDICTED_SALES']].values.tolist()

            HeatMap(heatmap_data, radius = 15, blur = 10, max_zoom = 1).add_to(hotspot_map)

            hotspot_map.save('interactive_hotspot_map.html')

            lm, rm = st.columns([6.5, 3.5])
            with lm:
                st.components.v1.html(hotspot_map._repr_html_(), height = 585)

            with rm:
                new_group = data.groupBy('CITY', 'SEASON', 'ITEM_CATEGORY', 'MENU_ITEM_NAME') \
                    .agg(F.avg('UNIT_PRICE').alias('UNIT_PRICE'),
                        F.avg('COST_OF_GOODS_USD').alias('AVG_COST'),
                        F.sum('PRICE').alias('TOTAL_SALES')) \
                    .withColumn('PROFIT', F.col('UNIT_PRICE') - F.col('AVG_COST')) \
                    .orderBy(F.desc('PROFIT'))

                new_group_pd = new_group.toPandas()

                def plot_sparkline(data):
                    """
                    Plots a sparkline showing seasonal sales trends.

                    Returns:
                        Figure: A Plotly figure with the seasonal sales sparkline.
                    """
                    
                    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
                    
                    data['SEASON'] = pd.Categorical(data['SEASON'], categories = season_order, ordered = True)
                    data = data.sort_values('SEASON')
                    
                    fig_spark = go.Figure(
                        data=go.Scatter(
                            x = data['SEASON'],  
                            y = data['TOTAL_SALES'],  
                            mode = 'lines',
                            fill = 'tozeroy',
                            line_color = 'red',
                            fillcolor = 'pink',
                        ),
                    )
                    
                    fig_spark.update_xaxes(
                        visible = True,  
                        tickvals = [season_order[0], season_order[-1]],  
                        ticktext = [season_order[0], season_order[-1]],  
                        fixedrange = True,
                    )
                    
                    fig_spark.update_yaxes(visible = False, fixedrange = True)
                    fig_spark.update_layout(
                        showlegend = False,
                        plot_bgcolor = 'white',
                        height = 60,
                        margin = dict(t = 10, l = 0, b = 0, r = 0, pad = 0),
                    )
                    return fig_spark

                def cards_with_season_city(menu_item, item_category, avg_cost, unit_price, profit, season_data, unique_id):
                    """
                    Displays a card with sales data, profit, and a seasonal sparkline.
                    """

                    with st.container(border=True):
                        tl, tr = st.columns([2, 1])  
                        bl, br = st.columns([1, 1])  

                        with tl:
                            st.markdown(f"**{menu_item}**")
                            st.markdown(f'Item Category: {item_category}')

                        with tr:
                            st.markdown(f"**Unit Price:** $ {unit_price:.2f}")
                            st.markdown(f"**Production Cost:** $ {avg_cost:.2f}")

                        with bl:
                            st.markdown(f"**Profit:**")
                            st.markdown(f"$ {profit:.2f}")

                        with br:
                            sparkline_chart = plot_sparkline(season_data)
                            st.plotly_chart(sparkline_chart, config = dict(displayModeBar = False), 
                                            use_container_width = True, key = f"sparkline_{unique_id}")

                def display_top_items_vertical(new_group_pd):
                    """
                    Displays top 3 menu items by total profit with sales trends.
                    """

                    result =  duckdb.query("""
                            SELECT DISTINCT MENU_ITEM_NAME,
                                    ITEM_CATEGORY,
                                    AVG(AVG_COST) AS AVG_COST,
                                    AVG(UNIT_PRICE) AS UNIT_PRICE,
                                    AVG(PROFIT) AS TOTAL_PROFIT,
                            FROM new_group_pd
                            GROUP BY MENU_ITEM_NAME, ITEM_CATEGORY
                            ORDER BY TOTAL_PROFIT DESC
                            LIMIT 3
                            """).df()

                    
                    for idx, row in result.iterrows():
                        menu_item = row['MENU_ITEM_NAME']
                        item_category = row['ITEM_CATEGORY']
                        avg_cost = row['AVG_COST']
                        unit_price = row['UNIT_PRICE']
                        profit = row['TOTAL_PROFIT']
                        
                        season_data = (
                            new_group_pd[new_group_pd['MENU_ITEM_NAME'] == menu_item]
                            .groupby('SEASON')['TOTAL_SALES']
                            .sum()
                            .reset_index()
                        )
                        
                        cards_with_season_city(
                            menu_item, item_category, avg_cost, unit_price, profit, season_data, unique_id=idx
                        )

                display_top_items_vertical(new_group_pd)

        else:
            st.write('No data found for the selected filters. Please try again with different options.')
    else:
        st.write("Please select filters and click 'Apply Selection' to view results.")
