#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pyodbc
import datetime as dt
import os

#set page configuration and title
st.set_page_config(page_title= 'Provider Review Portal',page_icon='🏥',layout='wide', initial_sidebar_state='expanded')

#assign the image file to a variable and display it
image = Image.open('tariff_portal_image.png')
st.image(image, use_column_width=True)

#query to import data from the DB and assign to a varaible as below
query = 'select * from [dbo].[tbl_ProviderReferralTariff]'

#a function to connect to the DB server, run the queries above and retrieve the data
@st.cache_data(ttl = dt.timedelta(hours=24))
def get_data_from_sql():
    server = os.environ.get('server_name')
    database = os.environ.get('db_name')
    username = os.environ.get('db_username')
    password = os.environ.get('password')
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER='
        + server
        +';DATABASE='
        + database
        +';UID='
        + username
        +';PWD='
        + password
        )
    # conn = pyodbc.connect(
    #     'DRIVER={ODBC Driver 17 for SQL Server};SERVER='
    #     +st.secrets['server']
    #     +';DATABASE='
    #     +st.secrets['database']
    #     +';UID='
    #     +st.secrets['username']
    #     +';PWD='
    #     +st.secrets['password']
    #     )
    provider_referal = pd.read_sql(query,conn)
    conn.close()
    return provider_referal
#use the function above to retrieve the data and assign to a varaible
referral_df = get_data_from_sql()
#convert the description column to upper case
referral_df['CPTDescription'] = referral_df['CPTDescription'].str.upper()

# #filter for all service categories except drugs and consumables
# referral_df = referral_df[
#     (referral_df['ServiceCategory'] == 'PROCEDURE') |
#     (referral_df['ServiceCategory'] == 'SERVICE') |
#     (referral_df['ServiceCategory'] == 'CONSULTATION')
#     ]
#create a list of unique services in the dataframe
unique_service = referral_df['CPTDescription'].unique()
#create a sidebar select box to select required service from the unique list of services above
selected_service = st.sidebar.selectbox(label='Select Service', options=unique_service)
#create a list of unique location states where the selected services are available
unique_loc = referral_df.loc[referral_df['CPTDescription'] == selected_service, 'State'].unique()
#create a sidebar selectbox that enables users to select the state
selected_loc = st.sidebar.selectbox(label='Select Location', options=unique_loc)
#create a list of unique class of provider offering the selected service in a particular State
unique_class = referral_df.loc[
    (referral_df['CPTDescription'] == selected_service) &
    (referral_df['State'] == selected_loc),
    'ProviderClass'
].unique()
#create a sidebar select box to select the required provider class
selected_class = st.sidebar.selectbox('Select ProviderClass', options=unique_class)
#condition to extract the standard tariff for the 5 different levels
if not referral_df.empty:
    #select only columns with the standard tariffs
    sel_cols = ['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5']
    #create a dataframe for only the standard tariffs for the selected service
    display_df = referral_df[referral_df['CPTDescription'] == selected_service][sel_cols]
    #set of instructions to be executed if dataframe is not empty
    if not display_df.empty:
        #extract the tariff for each level and assign to a variable
        level_1_amount = display_df['Level_1'].iloc[0]
        level_2_amount = display_df['Level_2'].iloc[0]
        level_3_amount = display_df['Level_3'].iloc[0]
        level_4_amount = display_df['Level_4'].iloc[0]
        level_5_amount = display_df['Level_5'].iloc[0]
        #add the # sign and a thousand seperator to the variable
        level_1_amount = '#' + '{:,}'.format(level_1_amount) 
        level_2_amount = '#' + '{:,}'.format(level_2_amount) 
        level_3_amount = '#' + '{:,}'.format(level_3_amount) 
        level_4_amount = '#' + '{:,}'.format(level_4_amount) 
        level_5_amount = '#' + '{:,}'.format(level_5_amount) 
        #display a title and the standard tariff for the 5 Avon tariff levels for the selected service
        st.subheader(f'AVON Standard Tariff for {selected_service}')
        st.write(f'Level 1::::     {level_1_amount}')
        st.write(f'Level 2::::     {level_2_amount}')
        st.write(f'Level 3::::     {level_3_amount}')
        st.write(f'Level 4::::     {level_4_amount}')
        st.write(f'Level 5::::     {level_5_amount}')

    else:
        st.write('Service not Available in AVON Standard Tariff')
else:
    st.write('DataFrame is empty or not available')

#filter for only providers offering the selected service in the selected state and provider class
sel_service_df = referral_df[
    (referral_df['CPTDescription'] == selected_service) &
    (referral_df['State'] == selected_loc) &
    (referral_df['ProviderClass'] == selected_class)
]
#select only certain columns to be displayed and sort the values by the amount the provider is charging for the service
sel_service_df = sel_service_df[['ProviderName','ProviderClass','StandardDescription','Amount', 'Address','HMOOfficerName','HMODeskPhoneNo','HMOOfficerEmail']].sort_values(by='Amount',ascending=False).reset_index(drop=True)
#rename the description column
sel_service_df.rename(columns={'StandardDescription':'ProviderDescription'}, inplace=True)
#drop duplicate providers
sel_service_df.drop_duplicates(subset=['ProviderName', 'ProviderDescription'])
#display a header and the list of providers based on the selected options
st.subheader(f'Table of Providers Tariff for {selected_service} in {selected_loc}')
st.write(sel_service_df)