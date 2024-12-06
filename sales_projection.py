import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, row_number
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import plotly.graph_objects as go
from prophet import Prophet
import plotly.express as px
import gdown


selected_columns = ['TRUCK_BRAND_NAME', 'MENU_TYPE', 'PRICE', 'CITY', 'ITEM_CATEGORY', 'ITEM_SUBCATEGORY', 'ORDER_TOTAL', 'ORDER_TS']

spark = SparkSession.builder \
    .appName('Truck_Data') \
    .getOrCreate()

url = "https://drive.google.com/uc?id=1_D6T_1hw_G4koH7YzLawoP_J6nn6zHEJ"
source_file = "truck_data.parquet"

# Download the dataset
gdown.download(url, source_file, quiet=False)

def load_sample_data(source_file, selected_columns):
    """
    Loads a sample of data and extracts unique values and date range.

    Returns:
        tuple: A tuple with unique truck brands, item categories, cities, and min/max years.
    """

    df_spark_sample = spark.read.parquet(source_file).select(selected_columns).sample(fraction=0.1, seed=17)
    
    df_spark_sample = df_spark_sample.withColumn('DATE', col('ORDER_TS').cast('date')) \
                       .withColumn('HOUR', hour(col('ORDER_TS'))) \
                       .drop('ORDER_TS')
    
    def get_unique_values(column_name):
        unique_values = [row[column_name] for row in df_spark_sample.select(column_name).distinct().collect()]
        return ['All'] + unique_values

    truck_brands = get_unique_values('TRUCK_BRAND_NAME')
    cities = get_unique_values('CITY')
    item_categories = get_unique_values('ITEM_CATEGORY')

    min_year = df_spark_sample.select(F.year(F.min('DATE')).alias('min_year')).collect()[0]['min_year']
    max_year = df_spark_sample.select(F.year(F.max('DATE')).alias('max_year')).collect()[0]['max_year']
        
    return truck_brands, item_categories, cities, min_year, max_year

truck_brands, categories, cities, min_year, max_year = load_sample_data(source_file, selected_columns)

@st.cache_resource
def load_data(source_file, selected_columns, truck_brand, city, category):
    """
    Loads and filters data based on selected criteria for truck brand, city, and category.

    Returns:
        DataFrame: A filtered Spark DataFrame based on the user inputs.
    """

    df_spark = spark.read.parquet(source_file).select(selected_columns)
    
    df_spark = df_spark.withColumn('DATE', col('ORDER_TS').cast('date')) \
                       .withColumn('HOUR', hour(col('ORDER_TS'))) \
                       .drop('ORDER_TS')
    
    if 'All' not in truck_brand:
        df_spark = df_spark.filter(F.col('TRUCK_BRAND_NAME').isin(truck_brand))
    if 'All' not in city:
        df_spark = df_spark.filter(F.col('CITY').isin(city))
    if 'All' not in category:
        df_spark = df_spark.filter(F.col('ITEM_CATEGORY').isin(category))

    return df_spark

def app():
    st.title('Explore Tasty Bites Sales Performance and Projections (Poland)')

    st.divider()

    st.sidebar.title('Filters')

    filtered_data = None

    selected_truck_brand = st.sidebar.selectbox('Select Truck Brand', options = truck_brands, index = 0)
    selected_city = st.sidebar.radio('Select City', options = cities, index = 0, horizontal = True)
    selected_categories = st.sidebar.multiselect('Select Item Category', options = categories, default = 'All')
    
    if min_year == max_year:
        st.sidebar.warning('Data is available only for the year: {}'.format(min_year))
        selected_year_range = (min_year, max_year)
    else:
        selected_year_range = st.sidebar.slider('Select Year Range', min_value = min_year, 
                                                max_value = max_year, value = (min_year, max_year)
                                                )

    df_spark = load_data(source_file, selected_columns, selected_truck_brand, selected_city, selected_categories)

    if st.sidebar.button('Apply Selection'):
        filtered_data = df_spark

    if filtered_data is not None:
        filtered_data = df_spark.filter(
                (F.year(F.col('DATE')) >= selected_year_range[0]) &
                (F.year(F.col('DATE')) <= selected_year_range[1])
            )


        summary_df = filtered_data.groupBy('ITEM_SUBCATEGORY') \
                                  .agg(
                                        F.avg('PRICE').alias('avg_spending'),
                                        F.sum('PRICE').alias('total_sales')
                                    )
        summary_df_pd = summary_df.toPandas()

        # Function to create sparkline plot
        def plot_sparkline(data):
            """
            Creates a sparkline plot for daily sales.

            Returns:
                Figure: A Plotly figure displaying the sales trend as a sparkline.
            """

            fig_spark = go.Figure(
                data=go.Scatter(
                    x = data['DATE'],  
                    y = data['daily_total'], 
                    mode = 'lines',
                    fill = 'tozeroy',
                    line_color = 'red',
                    fillcolor = 'pink',
                ),
            )
            fig_spark.update_traces(hovertemplate='Total Sales: $ %{y:.2f}')
            fig_spark.update_xaxes(visible = False, fixedrange = True)
            fig_spark.update_yaxes(visible = False, fixedrange = True)
            fig_spark.update_layout(
                showlegend=False,
                plot_bgcolor='white',
                height=50,
                margin=dict(t = 10, l = 0, b = 0, r = 0, pad = 0),
            )
            return fig_spark

        def cards(item_subcategory, total_sales, avg_spending, daily_total, option_type):
            """
            Displays a card with sales data, average spending, and a sparkline.
            """

            with st.container(border=True):
                tl, tr = st.columns([2, 1])
                bl, br = st.columns([1, 1])

                icons = {
                    'Hot Option': '🌶️',  
                    'Cold Option': '🍦', 
                    'Warm Option': '🍜'   
                }

                tl.markdown(f'**{icons.get(option_type, "")} {item_subcategory}**')

                tr.markdown(f'**Total Sales**')
                tr.markdown(f'$ {total_sales:,.0f}')    

                with bl:
                    st.markdown(f'**Average Spending**')
                    st.markdown(f'$ {avg_spending:,.2f}')

                with br:
                    fig_spark = plot_sparkline(daily_total)
                    st.plotly_chart(fig_spark, config=dict(displayModeBar = False), use_container_width = True)
 

        def display_metrics(summary_df_pd):
            """
            Displays metrics for top 3 item subcategories with total sales, spending, and a sparkline.
            """

            col1, col2, col3 = st.columns(3)

            for idx, (_, row) in enumerate(summary_df_pd.iterrows()):
                if idx >= 3: 
                    break
                item_subcategory = row['ITEM_SUBCATEGORY']
                total_sales = row['total_sales']
                avg_spending = row['avg_spending']

                if 'hot' in item_subcategory.lower():  
                    option_type = 'Hot Option'
                elif 'cold' in item_subcategory.lower(): 
                    option_type = 'Cold Option'
                else:
                    option_type = 'Warm Option' 

                daily_totals_subcategory = filtered_data.filter(F.col('ITEM_SUBCATEGORY') == item_subcategory) \
                    .groupBy('DATE').agg(F.sum('PRICE').alias('daily_total')) \
                    .orderBy('DATE') \
                    .toPandas()

                with [col1, col2, col3][idx]: 
                    cards(item_subcategory, total_sales, avg_spending, daily_totals_subcategory, option_type)
                    
        st.markdown('### The 3 Product Subcategories Sold by Trucks')
        display_metrics(summary_df_pd)

        df_grouped = filtered_data.groupBy('DATE', 'ITEM_CATEGORY').agg(F.sum('PRICE').alias('PRICE'))

        df_pivot = df_grouped.groupBy('DATE').pivot('ITEM_CATEGORY').sum('PRICE').fillna(0)

        df_pandas = df_pivot.toPandas()
        df_pandas = df_pandas.map(lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x)
        df_pandas['DATE'] = pd.to_datetime(df_pandas['DATE'])
        df_pandas = df_pandas.resample('W', on = 'DATE').sum().reset_index()
        df_pandas = df_pandas.iloc[:-1] 

        forecast_results = []
        for category in df_pandas.columns[1:]: 
            category_data = df_pandas[['DATE', category]].rename(columns = {category: 'PRICE'})
            category_data.columns = ['ds', 'y']
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality = False, daily_seasonality = False)
            model.fit(category_data)
            
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            forecast['ITEM_CATEGORY'] = category
            forecast['Type'] = 'Forecast'
            category_data['ITEM_CATEGORY'] = category
            category_data['Type'] = 'Actual'

            forecast_results.append(pd.concat([category_data, forecast[['ds', 'yhat', 'ITEM_CATEGORY', 'Type']].rename(columns={'yhat': 'y'})]))

        combined_forecast_df = pd.concat(forecast_results)

        fig = px.line(
            combined_forecast_df,
            x = 'ds',
            y = 'y',
            color = 'ITEM_CATEGORY',
            line_dash = 'Type',  
            labels = {'ds': 'Date', 'y': 'Price'}
        )

        fig.update_layout(
            xaxis_title = 'Date',
            yaxis_title = 'Total Sales',
            legend_title = 'Category & Type',
            height=650
        )

        l, r = st.columns([7.3, 2.7])
        l.markdown('### Sales Forecast for the next year with actual data')
        l.plotly_chart(fig)


        menu_data = [
            ("Vegetarian", 4.0, "https://www.gigadocs.com/blog/wp-content/uploads/2020/03/istock-955998758.jpg"),
            ("Sandwiches", 4.5, "https://realfood.tesco.com/media/images/RFO-MAIN-472x310-ChickenClubSandwich-4ec15d7a-9867-4b62-831d-62dd9ad5c039-0-472x310.jpg"),
            ("Poutine", 4.0, "https://images.themodernproper.com/billowy-turkey/production/posts/2022/HomemadePoutineRecipe_Shot8_78.jpg?w=960&h=720&q=82&fm=jpg&fit=crop&dm=1667570349&s=5114ab356816ecbca4ab2eff55ac4b13"),
            ("BBQ", 5.0, "https://www.hartsfurniture.co.uk/media/mageplaza/blog/post/5/_/5._bbq.jpg"),
            ("Mac & Cheese", 4.5, "https://www.allrecipes.com/thmb/e8uotDI18ieXNBY0KpmtGKbxMRM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/238691-Simple-Macaroni-And-Cheese-mfs_008-4x3-6ed91ba87a1344558aacc0f9ef0f4b41.jpg"),
            ("Ramen", 4.5, "https://www.kikkoman.co.uk/fileadmin/user_upload/kikkoman.eu/Food-News/UK_What-is-Ramen/Blog_What-is-Ramen_Header_Desktop.jpg"),
            ("Crepes", 4.0, "https://foodcrumbles.com/wp-content/uploads/2015/12/Frech-crepes-freshly-made-with-fruits-735x490.jpg"),
            ("Hot Dogs", 4.0, "https://staticcookist.akamaized.net/wp-content/uploads/sites/22/2022/07/Hot-dogs-10.jpg"),
            ("Tacos", 4.5, "https://gypsyplate.com/wp-content/uploads/2022/07/birria-tacos_square.jpg"),
            ("Ethiopian", 3.5, "https://migrationology.com/wp-content/uploads/2014/02/ethiopian-food.jpg"),
            ("Indian", 4.0, "https://www.tastingtable.com/img/gallery/20-delicious-indian-dishes-you-have-to-try-at-least-once/intro-1645057933.jpg"),
            ("Gyros", 4.0, "https://www.allincrete.com/wp-content/uploads/2019/12/two-lamb-gyros-with-feta-cheese-tzatziki-sauce-allincrete.com_.jpg"),
            ("Grilled Cheese", 4.5, "https://grilledcheesesocial.com/wp-content/uploads/2024/04/marry-me-grilled-cheese-sandwich-12.jpg"),
            ("Ice Cream", 5.0, "https://www.chhs.colostate.edu/fsi/wp-content/uploads/sites/51/2024/07/ice-cream-cones-1-800x0-c-default.jpg")
            ]

        columns = ['MENU_TYPE', 'RATING', 'IMAGE']

        menu_df = spark.createDataFrame(menu_data, schema=columns)

        reduced_df = filtered_data.select('ITEM_SUBCATEGORY', 'MENU_TYPE', 'PRICE')


        data = reduced_df.join(menu_df, on= 'MENU_TYPE', how = 'left')
        data = data.groupBy('ITEM_SUBCATEGORY', 'MENU_TYPE', 'IMAGE') \
                .agg(F.sum('PRICE').alias('TOTAL'),
                    F.avg('RATING').alias('RATING')
                    )
        
        data = data.toPandas()
        data = data.sort_values(by= ['ITEM_SUBCATEGORY', 'TOTAL'], ascending = [True, False])

        top_3 = data.groupby('ITEM_SUBCATEGORY').head(3).reset_index(drop=True)

        def display_top_3_by_category(data, category, title, r):
            """
            Displays top 3 menu items by category with sales, rating, and images.
            """
            
            filtered_data = data.query(f'ITEM_SUBCATEGORY == "{category}"').drop(columns='ITEM_SUBCATEGORY')
            
            # Render the markdown and data editor
            r.markdown(f'#### TOP 3 Menus in {title}')
            r.data_editor(
                filtered_data,
                column_config={
                    'IMAGE': st.column_config.ImageColumn(
                        'Image',
                        help='Menu Item Image'
                    ),
                    'MENU_TYPE': 'Menu Type',
                    'TOTAL': st.column_config.NumberColumn('Total Sales', format='$ %.0f'),
                    'RATING': st.column_config.NumberColumn(
                        'Rating',
                        help='Customer Rating (out of 5 stars)',
                        format='%d ⭐'
                    )
                },
                hide_index=True
            )

        display_top_3_by_category(top_3, "Hot Option", "Hot Option", r)
        display_top_3_by_category(top_3, "Warm Option", "Warm Option", r)
        display_top_3_by_category(top_3, "Cold Option", "Cold Option", r)
        # cold = top_3.query('ITEM_SUBCATEGORY == "Cold Option"')
        # cold = cold.drop(columns = 'ITEM_SUBCATEGORY')
        # warm = top_3.query('ITEM_SUBCATEGORY == "Warm Option"')
        # warm = warm.drop(columns = 'ITEM_SUBCATEGORY')
        # hot = top_3.query('ITEM_SUBCATEGORY == "Hot Option"')
        # hot = hot.drop(columns = 'ITEM_SUBCATEGORY')

        # r.markdown('#### TOP 3 Menus in Hot Option')
        # r.data_editor(
        #     hot,
        #     column_config={
        #         'IMAGE': st.column_config.ImageColumn(
        #         'Image',
        #         help = 'Menu Item Image'
        #         ),
        #         'ITEM_SUBCATEGORY': 'Subcategory',
        #         'MENU_TYPE': 'Menu Type',
        #         'TOTAL': st.column_config.NumberColumn("Total Sales", format = "$ %.0f"),
        #         'RATING': st.column_config.NumberColumn(
        #             'Rating',
        #             help = 'Customer Rating (out of 5 stars)',
        #             format = '%d ⭐'
        #         )
        #     },
        #     hide_index=True
        # )

        # r.markdown('#### TOP 3 Menus in Warm Option')
        # r.data_editor(
        #     warm,
        #     column_config={
        #         'IMAGE': st.column_config.ImageColumn(
        #         'Image',
        #         help = 'Menu Item Image'
        #         ),
        #         'ITEM_SUBCATEGORY': 'Subcategory',
        #         'MENU_TYPE': 'Menu Type',
        #         'TOTAL': st.column_config.NumberColumn('Total Sales', format = '$ %.0f'),
        #         'RATING': st.column_config.NumberColumn(
        #             'Rating',
        #             help = 'Rating out of 5 stars',
        #             format = '%d ⭐'
        #         )
        #     },
        #     hide_index = True
        # )

        # r.markdown('#### TOP 3 Menus in Cold Option')
        # r.data_editor(
        #     cold,
        #     column_config={
        #         'IMAGE': st.column_config.ImageColumn(
        #         'Image',
        #         help = 'Menu Item Image'
        #         ),
        #         'ITEM_SUBCATEGORY': 'Subcategory',
        #         'MENU_TYPE': 'Menu Type',
        #         'TOTAL': st.column_config.NumberColumn("Total Sales", format = "$ %.0f"),
        #         'RATING': st.column_config.NumberColumn(
        #             'Rating',
        #             help = 'Customer Rating (out of 5 stars)',
        #             format='%d ⭐'
        #         )
        #     },
        #     hide_index = True
        # )
    
    else:
        st.write("Please select filters and click 'Apply Selection' to view results.")
