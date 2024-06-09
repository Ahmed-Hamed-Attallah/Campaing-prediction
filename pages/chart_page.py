import streamlit as st
import plotly.express as px
import pandas as pd
import joblib as jo

st.set_page_config(
        layout = 'wide',
        page_title = 'DashBoard',
        page_icon= '📊'
)


df = pd.read_csv('Bank_data.csv')
df['duration'] = df['duration'].apply(lambda x : x/60)
def x(x):
    if x == 999 :
        return 0
    else :
        return x
df['pdays'] = df['pdays'].apply(x)
tab1 , tab2= st.tabs(['Catigorical charts' , 'Nomerical charts'])

with tab1 :
    st.markdown('<h3 style="text-align: center; color :#1450DB;">Catigorical charts</h3>', unsafe_allow_html=True)

    co1,co2,co3 = st.columns([1,0.25,1])
    with co1:
        X = st.selectbox('select column' , ['job', 'marital', 'education', 'default', 'housing', 'loan',
           'contact', 'month', 'day_of_week', 'poutcome'])
    with co3 :
        list1 = [None,'job', 'marital', 'education', 'default', 'housing', 'loan',
           'contact', 'month', 'day_of_week', 'poutcome']
        list1.pop(list1.index(X))
        Color = st.selectbox('select color' ,list1 )
        if Color == None :
            hist_title = f'histogram graph for {X} column'
            pie_title = f'pie chart for {X} column'
        else :
            hist_title = f'histogram graph for {X} and colored by {Color}'
            pie_title = f'pie chart for {X} and colored by {Color} '

    col1 , col2 , col3  = st.columns([1,0.25 , 1])
    with col1 :
        st.plotly_chart(px.histogram(df , x = X , color=Color ,title=hist_title,color_discrete_sequence=px.colors.qualitative.D3) , use_container_width=True)

    with col3 :
        st.plotly_chart(px.pie(df, names=X, color = Color,title=pie_title,color_discrete_sequence=px.colors.qualitative.D3, hole=0.3).update_traces(textinfo='value') , use_container_width=True)
    col11 , col22 , col33 = st.columns([1,0.25 , 1])
    with col11 :
        columns = st.multiselect('select one or more than' , ['job', 'marital', 'education', 'default', 'housing', 'loan',
           'contact', 'month', 'day_of_week', 'poutcome'])

    with col33 :
        color1 = st.selectbox('Select color' , [None,'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week', 'poutcome' ] )

    co11 , co22 , co33 = st.columns([1,0.25 , 1])
    if (type(columns) == list )and (color1 == None) :
            sun_title = f'Sunburst chart for {columns} columns'
            tree_title = f'treemap chart for {columns} columns'
    elif (type(columns) != list )and (color1 == None) :
        tree_title = f'treemap chart for {columns} column'
        sun_title = f'Sunburst chart for {columns} column'
    elif (type(columns) == list )and (color1 != None) :
        sun_title = f'Sunburst chart for all these columns {columns} and colored by {color1}'
        tree_title = f'treemap chart for all these columns {columns} and colored by {color1}'
    else :
        sun_title = f'Sunburst chart for {columns} and colored by {color1}'
        tree_title = f'treemap chart for {columns} and colored by {color1}'
    with co11 :
         st.plotly_chart(px.sunburst(df, path=columns,color =color1 ,title=sun_title, color_continuous_scale= px.colors.sequential.Brwnyl) , use_container_width=True)
    with co33 :
        st.plotly_chart(px.treemap(df, path=columns,color = color1 , title=tree_title,color_continuous_scale= px.colors.sequential.Mint),use_container_width=True)


with tab2 :
        st.markdown('<h3 style="text-align: center; color :#1450DB;">Nomerical charts</h3>', unsafe_allow_html=True)
        column = st.selectbox('select x from here :' , [ 'duration', 'campaign', 'pdays',
       'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed'])
        col1 , col2 , col3 = st.columns([1,0.25 , 1])
        with col1 :
            st.plotly_chart(px.histogram(df , x = column , title = f'histogram graph for {column} column',color_discrete_sequence=px.colors.sequential.Agsunset) , use_container_width=True)
        with col3 :
            st.plotly_chart(px.box(df , x = column , title = f'box plot for {column} column',color_discrete_sequence=px.colors.sequential.dense) , use_container_width=True)
        list1 = [None , 'duration', 'campaign', 'pdays','previous', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
        list1.pop(list1.index(column))
        column2 = st.selectbox('select y from here' ,list1)
        if column2 == None :
            scat_title = f'scatter plot for{column} column'
            line_title = f'line plot for {column} column'
        else :
            scat_title = f'scatter plot for {column} with {column2}'
            line_title = f'line plot for {column} with {column2}'
        st.plotly_chart(px.line(df.sort_values(by =column), x = column , y = column2 ,title=line_title,color_discrete_sequence=px.colors.qualitative.Pastel) , use_container_width=True)
        st.plotly_chart(px.scatter(df, x = column , y = column2 , title=scat_title , color_discrete_sequence=px.colors.carto.Blugrn) , use_container_width=True)
