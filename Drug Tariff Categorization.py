#Import necessary libraries
import pandas as pd
import streamlit as st
import datetime as dt
from rapidfuzz import fuzz
import pyodbc
import os

# #Get the screen image, assign to a variable and display the variable
# image = Image.open('tariff_portal_image.png')
# st.image(image, use_column_width=True)

# query = 'select * from [dbo].[tbl_AvonRevisedProposedStandardTariff]'
query1 = "select * from [dbo].[tbl_CurrentProviderTariff]\
            where cptcode not like 'NHIS%'\
                and ServiceCategory = 'Supply'"
query2 = 'select Code HospNo,\
        Name ProviderName,\
        ProviderClass,\
        Address,\
        State,\
        City,\
        PhoneNo,\
        Email,\
        ProviderManager,\
        ProviderGroup\
        from [dbo].[tbl_ProviderList_stg]'
query3 = 'select * from [dbo].[tbl_CPTCodeMaster]'
query4 = 'select * from [dbo].[tbl_CPTmappeddrugtariff]'

@st.cache_data(ttl = dt.timedelta(hours=24))
def get_data_from_sql(query_list):
    server = os.environ.get('server_name')
    database = os.environ.get('db_name')
    username = os.environ.get('db_username')
    password = os.environ.get('db_password')
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
    dfs = [pd.read_sql(q, conn) for q in query_list]
    conn.close()
    return dfs

#apply the function above and assign the imported data to variables
provider_tariff, provider_details, service_details,drug_tariff = get_data_from_sql([query1, query2, query3, query4])


#Rename column CPTDescription to ProvDescription
provider_tariff = provider_tariff.rename(columns={'CPTDescription': 'ProvDescription', 'cptcode': 'CPTCode'})

#Ensuring the CPTCode and CPTDescription columns below are converted to upper case for case sensitive joining purposes in lookup
provider_tariff['ProvDescription'] = provider_tariff['ProvDescription'].str.upper()
provider_tariff['CPTCode'] = provider_tariff['CPTCode'].str.upper()

drug_tariff['CPTCode'] = drug_tariff['CPTCode'].str.upper()
drug_tariff['CPTDescription'] = drug_tariff['CPTDescription'].str.upper()
service_details['StandardDescription'] = service_details['StandardDescription'].str.upper()

#Filter the provider_tariff and service_details dataframe to only supply
# provider_tariff = provider_tariff[provider_tariff['ServiceCategory'] == 'Supply']
service_details = service_details[service_details['ServiceType'] == 'SUPPLY']

#Merged the provider tariff and provider details dataframes and select the necessary columns needed
merged_provider_tariff = pd.merge(provider_tariff, provider_details, how='inner', on='HospNo', indicator='Exist')

merged_provider_tariff = merged_provider_tariff[['CPTCode', 'ProvDescription', 'Amount','ProviderName', 'ProviderClass', 'State', 'ProviderGroup']]



#Get variance difference of each provider tariff from the standard level tariff
#Function to get the percent change variance to be called when calculating
def percent_change2(col1,col2):
    return ((col1-col2)/col2)*100

#Then a function to compare the service description from provider with the standard description from AVON
#and assign a matching score to each services
def compare_cpt_description2(col3,col4):
    return fuzz.ratio(col3,col4)

#Time to merge the merged_provider_tariff and new tariff dataframe. select first the columns required from each of the dataframes
#columns to merge from merged_provider_tariff
cols_tomerge = ['CPTCode', 'ProvDescription', 'Amount', 'ProviderName','ProviderClass', 'State', 'ProviderGroup']
#columns to merge from new_tariff
cols_tomerge2 = ['CPTCode', 'CPTDescription', 'Level_1_Unit_price', 'Level_2_Unit_price', 'Level_3_Unit_price', 'Level_4_Unit_price', 'Level_5_Unit_price']

#Now, merge the two dataframes on CPTCode outside the loop for variance calculation
merged_provider_standard_tariff2 = pd.merge(merged_provider_tariff[cols_tomerge], drug_tariff[cols_tomerge2], how='inner', on='CPTCode', indicator='Exist')

#Let's get the variance difference % of each drug tariff from the 5 different standard level tariff and add as columns to the new dataframe
merged_provider_standard_tariff2['Tariff-L1%'] =round( percent_change2(merged_provider_standard_tariff2['Amount'], merged_provider_standard_tariff2['Level_1_Unit_price']))
merged_provider_standard_tariff2['Tariff-L2%'] =round( percent_change2(merged_provider_standard_tariff2['Amount'], merged_provider_standard_tariff2['Level_2_Unit_price']))
merged_provider_standard_tariff2['Tariff-L3%'] =round( percent_change2(merged_provider_standard_tariff2['Amount'], merged_provider_standard_tariff2['Level_3_Unit_price']))
merged_provider_standard_tariff2['Tariff-L4%'] =round( percent_change2(merged_provider_standard_tariff2['Amount'], merged_provider_standard_tariff2['Level_4_Unit_price']))
merged_provider_standard_tariff2['Tariff-L5%'] =round( percent_change2(merged_provider_standard_tariff2['Amount'], merged_provider_standard_tariff2['Level_5_Unit_price']))


#Rename columns for differences in provider and standard decription names
merged_provider_standard_tariff2.rename(columns={'ProvDescription':'ProviderServiceDescription', 'CPTDescription':'AVONStandardDescription'}, inplace=True)
#Create a new column that gives the matching score to each service
merged_provider_standard_tariff2['Match_score']= merged_provider_standard_tariff2.apply(lambda row: compare_cpt_description2(row['ProviderServiceDescription'], row['AVONStandardDescription']), axis=1)
#the full columns are listed below for the new dataframe
merged_provider_standard_tariff2 =merged_provider_standard_tariff2[['ProviderClass', 'ProviderName', 'State', 'ProviderGroup', 'CPTCode', 'ProviderServiceDescription','AVONStandardDescription',
                                                                        'Match_score', 'Amount','Level_1_Unit_price', 'Level_2_Unit_price', 'Level_3_Unit_price', 'Level_4_Unit_price', 'Level_5_Unit_price',
                                                                          'Tariff-L1%', 'Tariff-L2%', 'Tariff-L3%', 'Tariff-L4%', 'Tariff-L5%']]

#Select the display columns
select_providerdf2 = merged_provider_standard_tariff2[['CPTCode', 'ProviderServiceDescription', 'AVONStandardDescription','Match_score', 'Amount']]

def display_data(df):
    tariff_level = st.sidebar.selectbox(label='Recommended Tariff Level', options=['All', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6'])
    if tariff_level == 'All':
        data = df
    elif tariff_level == 'Level 1':
        data = df[df['Recommendation'] == 'Level 1'].reset_index(drop=True)
    elif tariff_level == 'Level 2':
        data = df[df['Recommendation'] == 'Level 2'].reset_index(drop=True)
    elif tariff_level == 'Level 3':
        data = df[df['Recommendation'] == 'Level 3'].reset_index(drop=True)
    elif tariff_level == 'Level 4':
        data = df[df['Recommendation'] == 'Level 4'].reset_index(drop=True)
    elif tariff_level == 'Level 5':
        data = df[df['Recommendation'] == 'Level 5'].reset_index(drop=True)
    elif tariff_level == 'Level 6':
        data = df[df['Recommendation'] == 'Level 6'].reset_index(drop=True)
    return data


#Function to get the average varaince difference from merged_provider_standard_tariff2 dataframe across all services for each provider
def aggregate_providertariff(providercategory, tarifflevel):
    #Filters the merged_provider_standard_tariff2 DataFrame to include only rows where ProviderClass matches the given providercategory
    #.copy() is used to avoid modifying the original DataFrame.
    combined_data = merged_provider_standard_tariff2[merged_provider_standard_tariff2['ProviderClass'] == providercategory].copy()
    # st.write(combined_data.head())
    #Add column "Variance" to store the % difference between Provider Amount and tariff level from drug_tariff dataframe using percent_change2 function
    combined_data['Variance'] = round(percent_change2(combined_data['Amount'],combined_data[tarifflevel]),2)

    #Get Provider, State dataframe by grouping
    providerstate = combined_data[['ProviderName', 'State']].drop_duplicates()

    #Group the dataframe by ProviderName
    groupeddata = combined_data.groupby(['ProviderName'])

    # st.write(groupeddata.head())
    #Each line here calculates the average variance from a standard tariff level for every provider:
    df_cond = groupeddata.apply(lambda x: round(x['Variance'].mean(),2)).reset_index(name = 'Average_variance')
    df_L1 = groupeddata.apply(lambda x: round(x['Tariff-L1%'].mean(),2)).reset_index(name = 'L1average')
    df_L2 = groupeddata.apply(lambda x: round(x['Tariff-L2%'].mean(),2)).reset_index(name = 'L2average')
    df_L3 = groupeddata.apply(lambda x: round(x['Tariff-L3%'].mean(),2)).reset_index(name = 'L3average')
    df_L4 = groupeddata.apply(lambda x: round(x['Tariff-L4%'].mean(),2)).reset_index(name = 'L4average')
    df_L5 = groupeddata.apply(lambda x: round(x['Tariff-L5%'].mean(),2)).reset_index(name = 'L5average')

    #All the above averages are merged into a single DataFrame on ProviderName:
    combined_df = pd.merge(df_cond, df_L1, on='ProviderName')
    combined_df = pd.merge(combined_df, df_L2, on='ProviderName')
    combined_df = pd.merge(combined_df,df_L3, on='ProviderName')
    combined_df = pd.merge(combined_df,df_L4, on='ProviderName')
    combined_df = pd.merge(combined_df, df_L5, on='ProviderName')
    combined_df = pd.merge(combined_df, providerstate, on='ProviderName')
    combined_df = combined_df.fillna(0)


    #This function determines the recommended tariff level (Level 1 to Level 6) for each provider checking against threshold
    def recommendationfunction(row):
        unique_state = ['LAGOS','ABUJA', 'RIVERS']
        threshold =50 if row['State'] in unique_state else 25
        if row['L1average'] <= threshold: 
            return 'LEVEL 1'
        elif row['L2average'] <= threshold: 
            return 'LEVEL 2'
        elif row['L3average'] <= threshold: 
            return 'LEVEL 3'
        elif row['L4average'] <= threshold: 
            return 'LEVEL 4'
        elif row['L5average'] <= 100: 
            return 'LEVEL 5'
        else:
            return 'LEVEL 6'
    combined_df['Recommendation'] = combined_df.apply(recommendationfunction, axis=1)
    combined_df = combined_df[['ProviderName', 'L1average','L2average','L3average','L4average','L5average', 'Recommendation']]
    return combined_df


#Applying the aggregate_providertariff function to each leve, creating a new column in the new dataframe indicating level category on TOSHFA
Level1_providerdf = aggregate_providertariff('LEVEL 1', 'Level_1_Unit_price')
Level1_providerdf['TOSHFA Level'] = 'Level 1'
Level2_providerdf = aggregate_providertariff('LEVEL 2', 'Level_2_Unit_price')
Level2_providerdf['TOSHFA Level'] = 'Level 2'
Level3_providerdf = aggregate_providertariff('LEVEL 3', 'Level_3_Unit_price')
Level3_providerdf['TOSHFA Level'] = 'Level 3'
Level4_providerdf = aggregate_providertariff('LEVEL 4', 'Level_4_Unit_price')
Level4_providerdf['TOSHFA Level'] = 'Level 4'
Level5_providerdf = aggregate_providertariff('LEVEL 5', 'Level_5_Unit_price')
Level5_providerdf['TOSHFA Level'] ='Level 5'
Level6_providerdf = aggregate_providertariff('LEVEL 6', 'Level_5_Unit_price')
Level6_providerdf['TOSHFA Level'] ='Level 6'
allproviders =pd.concat([Level1_providerdf,Level2_providerdf,Level3_providerdf,Level4_providerdf,Level5_providerdf, Level6_providerdf])
allproviders = allproviders[['ProviderName', 'TOSHFA Level', 'Recommendation']]

# #Display the sub header for Provider Recommendation table
# st.subheader(f'Provider Categoty Recommendation based on Drug Tariff against Standard Levels Tariff')
# #Display the data
# st.write(allproviders)
# #Get the rows and columns count for the new dataframe
# allprovidersshape = allproviders.shape
# #Display the distinct provider count
# st.write(f'The provider Recommendation table contains {allprovidersshape[0]} distinct providers')

def calculate_rec(df, provider, location):
    """
    Calculate the average variance for each tariff level and determine a recommendation 
    based on variance thresholds and location-specific conditions.
    """
    # Thresholds based on location
    location_threshold = 50 if location in ["LAGOS", "ABUJA", "RIVERS"] else 25
    level_5_threshold = 100  # Additional condition for Level 5 before Level 6 categorization

    # Calculate the average variance for each tariff level
    variance_averages = {
        f'L{i}_ave': round(df[f'Tariff-L{i}%'].mean(), 2)
        for i in range(1, 6)
    }

    # Create a DataFrame to summarize results
    data = {
        'Condition': ['Overall'],
        **{f'Level {i} Variance': [variance_averages[f'L{i}_ave']] for i in range(1, 6)},
    }
    
    table_df = pd.DataFrame(data)

    # Ensure the data in the table contains only unique records
    table_df = table_df.drop_duplicates()

    # Recommendations for each level
    recommendations = {
        f'L{i}_rec': (
            f"The Drug Tariff of {provider} has a variance of {variance_averages[f'L{i}_ave']}% from "
            f"Standard LEVEL {i} Tariff and is hereby recommended to TARIFF LEVEL {i}."
        )
        for i in range(1, 6)
    }

       # Logic to determine recommendation for Levels 1–4
    for i in range(1, 5):
        if variance_averages[f'L{i}_ave'] <= location_threshold:
            return table_df, recommendations[f'L{i}_rec']
    
    # Handle Level 5 explicitly
    if variance_averages['L5_ave'] <= level_5_threshold:
        return table_df, recommendations['L5_rec']

    # If Level 5 variance exceeds level_5_threshold, recommend Level 6
    if variance_averages['L5_ave'] > level_5_threshold:
        rec_level_6 = (
            f"The Drug Tariff of {provider} has a variance of {variance_averages['L5_ave']}% "
            "on LEVEL 5 and is hereby recommended to TARIFF LEVEL 6."
        )
        return table_df, rec_level_6

    # Fallback (should rarely happen now)
    fallback_rec = (
        f"The Drug Tariff of {provider} does not meet any of the thresholds for recommendation "
        "to a specific TARIFF LEVEL based on the current variance analysis."
    )
    return table_df, fallback_rec


def display_data(df):
    tariff_level = st.sidebar.selectbox(label='Recommended Tariff Level', options=['All', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6'])
    if tariff_level == 'All':
        data = df
    elif tariff_level == 'Level 1':
        data = df[df['Recommendation'] == 'Level 1'].reset_index(drop=True)
    elif tariff_level == 'Level 2':
        data = df[df['Recommendation'] == 'Level 2'].reset_index(drop=True)
    elif tariff_level == 'Level 3':
        data = df[df['Recommendation'] == 'Level 3'].reset_index(drop=True)
    elif tariff_level == 'Level 4':
        data = df[df['Recommendation'] == 'Level 4'].reset_index(drop=True)
    elif tariff_level == 'Level 5':
        data = df[df['Recommendation'] == 'Level 5'].reset_index(drop=True)
    elif tariff_level == 'Level 6':
        data = df[df['Recommendation'] == 'Level 6'].reset_index(drop=True)
    return data

def display_provider_data(provider_class, provider_df, unique_providers):
    """Handles data display for a given provider class."""
    provider = st.sidebar.selectbox(label='Select Provider', options=unique_providers)
    location = provider_details.loc[provider_details['ProviderName'] == provider, 'State'].values[0]
    
    st.subheader('Summary of Recommended Level and Count of Providers')
    level_agg = provider_df.groupby('Recommendation').agg(ProviderCount=('Recommendation', 'count')).reset_index().sort_values(by='Recommendation', ascending=False)
    st.dataframe(level_agg)
    
    st.subheader(f'Recommended Tariff Level for {provider_class} Providers')
    st.dataframe(provider_df)
    
    # Filter and display detailed tariff information for the selected provider
    selected_provider_df = merged_provider_standard_tariff2[merged_provider_standard_tariff2['ProviderName'] == provider].reset_index(drop=True)
    selected_provider_df = selected_provider_df[['CPTCode', 'ProviderServiceDescription', 'AVONStandardDescription', 'Match_score', 'Amount',
                                                 'Level_1_Unit_price', 'Tariff-L1%', 'Level_2_Unit_price', 'Tariff-L2%', 'Level_3_Unit_price',
                                                 'Tariff-L3%', 'Level_4_Unit_price', 'Tariff-L4%', 'Level_5_Unit_price', 'Tariff-L5%']]
    #drop duplicates from selected_provider_df
    selected_provider_df = selected_provider_df.drop_duplicates(subset=['CPTCode', 'ProviderServiceDescription']).reset_index(drop=True)
    
    var_df, rec = calculate_rec(selected_provider_df, provider, location)
    
    st.subheader(f'Service Tariff Table for {provider}')
    st.dataframe(selected_provider_df)
    st.subheader(f'{provider} Service Tariff Variance from each Standard Tariff Level')
    st.dataframe(var_df)
    st.header('RECOMMENDATION')
    st.write(rec)

    return provider, selected_provider_df

select_task = st.sidebar.radio(
    label='Select Task', options=['New Tariff Review', 'Existing TOSHFA Tariff Review'])

if select_task == 'New Tariff Review':

    provider = st.sidebar.text_input('Type in Provider Name')
    location = st.sidebar.selectbox('Provider Location*', placeholder='Select Location', index=None, options=['Abia', 'Abuja', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 
                                                                                                        'Borno', 'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo',
                                                                                                        'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara','Lagos', 'Nasarawa',
                                                                                                        'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba',
                                                                                                        'Yobe', 'Zamfara'])
    #create a dictionary to map the uploaded file headers to a preferred name according to their index
    preffered_headers = {
        0: 'CPTCode',
        1: 'Description',
        2: 'ProviderTariff'
    }
    #add an uploader that enable users to upload provider tariff in uploadable format
    uploaded_file = st.sidebar.file_uploader('Upload the Provider Drug Tariff file already Mapped to CPT Codes here', type='csv')

    #set of instructions to be executed when a file is uploaded
    if location and uploaded_file:

        #include a select box on the sidebar that enables multiple selections to enable users to select multiple service category and frequency
        # service_cat = st.sidebar.multiselect('Select Service Category', ['DRUGS AND CONSUMABLES', 'CONSULTATIONS', 'INVESTIGATIONS', 'PROCEDURES', 'ROOMS AND FEEDING'])
        # frequency = st.sidebar.multiselect('Select Service Frequency', [5, 4, 3, 2, 1])
    #read the uploaded tariff into a pandas dataframe and assign to tariff
        tariff = pd.read_csv(uploaded_file, header=None, skiprows=1)

        #rename the columns based on the preferred_headers dictionary using index
        tariff.rename(columns=preffered_headers, inplace=True)

        #enforce the CPTCode columns to str
        tariff['CPTCode'] = tariff['CPTCode'].str.upper()

        #merge the provider tariff with the AVON standard tariff on CPTCode
        available_df = pd.merge(tariff, drug_tariff, on=['CPTCode'], how='inner', indicator='Exist')

    
        #available_df['Exist'] = np.where(available_df.Exist == 'both', True, False)
        #ensure the dataframe only returns records where the ProviderTariff > 0
        available_df['ProviderTariff'] = pd.to_numeric(available_df['ProviderTariff'], errors='coerce')
        available_df = available_df[available_df['ProviderTariff'] > 0]
        
        #change the description columns to uppercase
        available_df['Description'] = available_df['Description'].str.upper()
        available_df['CPTDescription'] = available_df['CPTDescription'].str.upper()
        #apply the first fuzzy function that compares the description columns and assign a score based on their compatibility to create a new column
        available_df['Match_Score'] = available_df.apply(lambda row: compare_cpt_description2(row['Description'], row['CPTDescription']), axis=1)
        #rename certain columns as below
        available_df.rename(columns={'Description':'ProviderDescription', 'CPTDescription':'StandardDescription'}, inplace=True)
        #return certain columns as selected below
        available = available_df[['CPTCode','ProviderDescription', 'StandardDescription','ProviderTariff','Match_Score']]
        

        #create new columns by applying the percent_change function to get the provider tariff variance from each standard tariff level
        available_df['Tariff-L1(%)'] = round(percent_change2(available_df['ProviderTariff'], available_df['Level_1_Unit_price']),2)
        available_df['Tariff-L2(%)'] = round(percent_change2(available_df['ProviderTariff'], available_df['Level_2_Unit_price']),2)
        available_df['Tariff-L3(%)'] = round(percent_change2(available_df['ProviderTariff'], available_df['Level_3_Unit_price']),2)
        available_df['Tariff-L4(%)'] = round(percent_change2(available_df['ProviderTariff'], available_df['Level_4_Unit_price']),2)
        available_df['Tariff-L5(%)'] = round(percent_change2(available_df['ProviderTariff'], available_df['Level_5_Unit_price']),2)

        # #create a list with the service categories to be used for recommendation
        # cat_for_rec = ['CONSULTATIONS', 'INVESTIGATIONS', 'PROCEDURES', 'ROOMS AND FEEDING']

        #function to calculate the average variance of the provider from the different standard tariff level based on the service frequency
        def calc_ave_var(lev_var):
            if available_df.empty:
                return 0
            
            # Calculate the mean
            ave_for_rec = available_df[lev_var].mean()
            
            # Return the rounded mean or 0 if ave_for_rec is None
            return round(ave_for_rec, 2) if ave_for_rec is not None else 0

        ave_for_rec_L1 = calc_ave_var('Tariff-L1(%)')
        ave_for_rec_L2 = calc_ave_var('Tariff-L2(%)')
        ave_for_rec_L3 = calc_ave_var('Tariff-L3(%)')
        ave_for_rec_L4 = calc_ave_var('Tariff-L4(%)')
        ave_for_rec_L5 = calc_ave_var('Tariff-L5(%)')

        # #write all the possible recommendations based on the results above and assign each recommendation to a variable
        rec1 = f'The Service Tariff of {provider} has a variance of {ave_for_rec_L1}% from Standard LEVEL 1 Tariff and is hereby recommended to TARIFF LEVEL 1'
        rec2 = f'The Service Tariff of {provider} has a variance of {ave_for_rec_L2}% from Standard LEVEL 2 Tariff and is hereby recommended to TARIFF LEVEL 2'
        rec3 = f'The Service Tariff of {provider} has a variance of {ave_for_rec_L3}% from Standard LEVEL 3 Tariff and is hereby recommended to TARIFF LEVEL 3'
        rec4 = f'The Service Tariff of {provider} has a variance of {ave_for_rec_L4}% from Standard LEVEL 4 Tariff and is hereby recommended to TARIFF LEVEL 4'
        rec5 = f'The Service Tariff of {provider} has a variance of {ave_for_rec_L5}% from Standard LEVEL 5 Tariff and is hereby recommended to TARIFF LEVEL 5'
        rec6 = f'The Service Tariff of {provider} has a variance of {ave_for_rec_L5}% from Standard LEVEL 5 Tariff and is hereby recommended to TARIFF LEVEL 6'
        

        #a function to assign a recommendation to the uploaded provider based on our logic and return the recommendation.
        def check_recommendation():
            # Define thresholds based on location
            threshold = 50 if location in ['Lagos', 'Abuja', 'Rivers'] else 25

            # Check recommendations
            recommendations = [rec1, rec2, rec3, rec4, rec5]
            averages = [ave_for_rec_L1, ave_for_rec_L2, ave_for_rec_L3, ave_for_rec_L4, ave_for_rec_L5]

            for avg, rec in zip(averages, recommendations):
                if avg <= threshold:
                    return rec

            # Default recommendation
            return rec6     

        filtered_df = available_df

        #another condition to filter the final table to be displayed based on the recommendation of the model for the provider
        #table to be displayed should contain the tariff level of the recommended level and a level below the recommended level
        if check_recommendation() == rec1:
            final_display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score', 'ProviderTariff', 'Level_1_Unit_price', 'Tariff-L1(%)']]
        elif check_recommendation() == rec2:
            final_display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score', 'ProviderTariff', 'Level_1_Unit_price', 'Tariff-L1(%)','Level_2_Unit_price', 'Tariff-L2(%)']]
        elif check_recommendation() == rec3:
            final_display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score', 'ProviderTariff', 'Level_2_Unit_price', 'Tariff-L2(%)', 'Level_3_Unit_price', 'Tariff-L3(%)']]
        elif check_recommendation() == rec4:
            final_display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score', 'ProviderTariff', 'Level_3_Unit_price', 'Tariff-L3(%)', 'Level_4_Unit_price', 'Tariff-L4(%)']]
        elif check_recommendation() == rec5:
            final_display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score', 'ProviderTariff', 'Level_4_Unit_price', 'Tariff-L4(%)', 'Level_5_Unit_price', 'Tariff-L5(%)']]
        elif check_recommendation() == rec6:
            final_display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score', 'ProviderTariff', 'Level_5_Unit_price', 'Tariff-L5(%)']]

        #calculate the average variance of the provider tariff from the standard levels based on the selected service category and frequency
        ave_var_L1 = round(filtered_df['Tariff-L1(%)'].mean(),2)
        ave_var_L2 = round(filtered_df['Tariff-L2(%)'].mean(),2)
        ave_var_L3 = round(filtered_df['Tariff-L3(%)'].mean(),2)
        ave_var_L4 = round(filtered_df['Tariff-L4(%)'].mean(),2)
        ave_var_L5 = round(filtered_df['Tariff-L5(%)'].mean(),2)

        #display a title for the uploaded provider services and classification
        st.title(provider + ' Services Available on AVON STANDARD TARIFF')
        #display only certain columns based on the selected columns below
        display_df = filtered_df[['CPTCode', 'ProviderDescription','StandardDescription','Match_Score','ProviderTariff','Level_1_Unit_price','Tariff-L1(%)','Level_2_Unit_price','Tariff-L2(%)', 'Level_3_Unit_price','Tariff-L3(%)', 'Level_4_Unit_price','Tariff-L4(%)', 'Level_5_Unit_price','Tariff-L5(%)']]
        #display the final_display_df 
        st.write(final_display_df)

        #display a title for the displayed variance based on the selections
        st.header('VARIANCE BASED ON SELECTIONS')
        st.write(f'The Average Tariff Variance of {provider} from Standard LEVEL 1 Tariff : {ave_var_L1}%')
        st.write(f'The Average Tariff Variance of {provider} from Standard LEVEL 2 Tariff : {ave_var_L2}%')
        st.write(f'The Average Tariff Variance of {provider} from Standard LEVEL 3 Tariff : {ave_var_L3}%')
        st.write(f'The Average Tariff Variance of {provider} from Standard LEVEL 4 Tariff : {ave_var_L4}%')
        st.write(f'The Average Tariff Variance of {provider} from Standard LEVEL 5 Tariff : {ave_var_L5}%')

        # Check the recommendation using the function
        recommendation = check_recommendation()
        st.header('RECOMMENDATION')
        #use markdown to style the recommendation
        st.markdown(
        f"""
        <style>
        .color-box {{
            background-color: #e3c062;
            padding: 15px;
            border-radius: 10px;
        }}
        </style>
        <div class='color-box'>{recommendation}</div>""",
        unsafe_allow_html=True,
    )
    else:
        st.error('Please select provider location to proceed')

elif select_task == 'Existing TOSHFA Tariff Review':
# Sidebar to select provider class
    provider_class = st.sidebar.selectbox(
        label='Select Current Provider Class', 
        options=['ALL', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6']
    )
    # Filter the provider dataset based on the selected class
    if provider_class == 'ALL':
        provider_df = display_data(allproviders)
        unique_providers = merged_provider_standard_tariff2['ProviderName'].unique()
    else:
        provider_df = display_data(eval(f"{provider_class.replace(' ', '')}_providerdf"))
        unique_providers = merged_provider_standard_tariff2.loc[
            merged_provider_standard_tariff2['ProviderClass'] == provider_class.upper(), 
            'ProviderName'
        ].unique()

    # Display data and perform analysis for the selected class
    provider, selected_provider_df = display_provider_data(provider_class, provider_df, unique_providers)