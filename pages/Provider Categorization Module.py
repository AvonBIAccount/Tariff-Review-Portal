#import needed libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime as dt
from fuzzywuzzy import fuzz

#open the image, assign to a variable and display the variable
image = Image.open('tariff_portal_image.png')
st.image(image, use_column_width=True)

#initialise the dataframe from session_state and assign to a variable to be used in this page
# standard_tariff = st.session_state['standard_tariff']
provider_tariff = st.session_state['provider_tariff']
provider_details = st.session_state['provider_details']
service_details = st.session_state['service_details']
new_tariff = st.session_state['new_tariff']

#merge the provider_tariff dataframe with provider_details dataframe, rename the cptcode column and select required columns
merged_provider_tariff = pd.merge(provider_tariff, provider_details, on=['HospNo'], how='inner', indicator='Exist')
merged_provider_tariff.rename(columns={'cptcode':'CPTCode'}, inplace=True)
merged_provider_tariff = merged_provider_tariff[['CPTCode', 'CPTDescription', 'Amount', 'ProviderName', 'ProviderClass', 'State', 'ProviderGroup']]
# merged_provider_tariff['HospNo'] = merged_provider_tariff['HospNo'].astype(str)

#function to calculate the percentage difference between levels.
def percent_change(col1, col2):
    return ((col1 - col2)/col2) * 100

#function to compare the service description of the provider with our cpt description and assign a matching score.
def compare_cpt_description(col1,col2):
    return fuzz.ratio(col1, col2)

#columns to merge from the merged_provider_tariff
cols_to_merge1 = ['CPTCode', 'CPTDescription', 'Amount', 'ProviderName', 'ProviderClass', 'State', 'ProviderGroup']
#columns to merge from the AVON standard tariff dataframe
cols_to_merge = ['CPTCode','CPTDESCRIPTION','Category', 'Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5']

# Merge standard_tariff outside the loop
merged_provider_standard_tariff = pd.merge(merged_provider_tariff[cols_to_merge1], new_tariff[cols_to_merge], on=['CPTCode'], how='left', indicator='Exist')


# Calculate %Variance of each service tariff from the 5 different levels and add as columns to the df
merged_provider_standard_tariff['Tariff-L1%'] = round(percent_change(merged_provider_standard_tariff['Amount'], merged_provider_standard_tariff['Level_1']), 2)
merged_provider_standard_tariff['Tariff-L2%'] = round(percent_change(merged_provider_standard_tariff['Amount'], merged_provider_standard_tariff['Level_2']), 2)
merged_provider_standard_tariff['Tariff-L3%'] = round(percent_change(merged_provider_standard_tariff['Amount'], merged_provider_standard_tariff['Level_3']), 2)
merged_provider_standard_tariff['Tariff-L4%'] = round(percent_change(merged_provider_standard_tariff['Amount'], merged_provider_standard_tariff['Level_4']), 2)
merged_provider_standard_tariff['Tariff-L5%'] = round(percent_change(merged_provider_standard_tariff['Amount'], merged_provider_standard_tariff['Level_5']), 2)


#Filter the dataframe for only certain categories and service frequency as shown below
merged_provider_standard_tariff = merged_provider_standard_tariff[
        (merged_provider_standard_tariff['Category'].isin(['Consultation', 'Procedure', 'Rooms and Feeding', 'Service']))
        # (merged_provider_standard_tariff['Frequency'].isin([5, 4, 3]))
    ]

#rename the description columns as below
merged_provider_standard_tariff.rename(columns = {'CPTDescription':'ProviderServiceDesc','CPTDESCRIPTION':'StandardServiceDesc'}, inplace=True)

#ensure the two description columns are changed to upper case
merged_provider_standard_tariff['ProviderServiceDesc'] = merged_provider_standard_tariff['ProviderServiceDesc'].str.upper()
merged_provider_standard_tariff['StandardServiceDesc'] = merged_provider_standard_tariff['StandardServiceDesc'].str.upper()
#create a new column that uses the fuzzy function above to compare the 2 service description and assign a score
merged_provider_standard_tariff['Match_Score'] = merged_provider_standard_tariff.apply(lambda row: compare_cpt_description(row['ProviderServiceDesc'], row['StandardServiceDesc']), axis=1)
#select required columns as selected below
merged_provider_standard_tariff = merged_provider_standard_tariff[['ProviderClass', 'ProviderName', 'State', 'ProviderGroup', 'Category', 'CPTCode', 'ProviderServiceDesc','StandardServiceDesc',
                                                                        'Match_Score', 'Amount', 'Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5',
                                                                          'Tariff-L1%', 'Tariff-L2%', 'Tariff-L3%', 'Tariff-L4%', 'Tariff-L5%']]

#function to aggregate the dataframe on a provider-by-provider basis
def aggregate_provider_tariff(providercategory, tariff_level):
    # Filter merged_provider_standard_tariff by based on the selected provider category
    combined_data = merged_provider_standard_tariff[merged_provider_standard_tariff['ProviderClass'] == providercategory].copy()
    #add a new column that calculates the %variance of the provider tariff using the percent_change function from the selected tariff_level
    combined_data['Variance'] = round(percent_change(combined_data['Amount'], combined_data[tariff_level]), 2)

    # extract the state for each provider
    provider_state = combined_data[['ProviderName', 'State']].drop_duplicates()

    # Group  the combined data by 'ProviderName'
    grouped_data = combined_data.groupby(['ProviderName'])

# st.write(grouped_data.head(1000))

    df_cond1 = grouped_data.apply(lambda x: round(x['Variance'].mean(), 2)).reset_index(name='Average Variance')
    df_L1 = grouped_data.apply(lambda x: round(x['Tariff-L1%'].mean(), 2)).reset_index(name='L1_average')
    df_L2 = grouped_data.apply(lambda x: round(x['Tariff-L2%'].mean(), 2)).reset_index(name='L2_average')
    df_L3 = grouped_data.apply(lambda x: round(x['Tariff-L3%'].mean(), 2)).reset_index(name='L3_average')
    df_L4 = grouped_data.apply(lambda x: round(x['Tariff-L4%'].mean(), 2)).reset_index(name='L4_average')
    df_L5 = grouped_data.apply(lambda x: round(x['Tariff-L5%'].mean(), 2)).reset_index(name='L5_average')

    #Merge all the dataframes above containing the average %variance of each provider from the 5 different tariff levels based on the 2 different conditions
    combined_df = pd.merge(df_cond1, df_L1, on='ProviderName')
    combined_df = pd.merge(combined_df,df_L2, on= 'ProviderName')
    combined_df = pd.merge(combined_df, df_L3, on='ProviderName')
    combined_df = pd.merge(combined_df, df_L4, on='ProviderName')
    combined_df = pd.merge(combined_df, df_L5, on='ProviderName')
    combined_df = pd.merge(combined_df, provider_state, on='ProviderName', how='left')

# Handle null values in combined_df
    combined_df = combined_df.fillna(0)

#  function that makes the recommendation for each provider based on the defined logic in the conditional statement below
    def calculate_rec(row):
        special_state = ['LAGOS', 'ABUJA', 'RIVERS']
        threshhold = 50 if row['State'] in special_state else 25
        if row['L1_average'] <= threshhold:
            return 'Level 1'
        elif row['L2_average'] <= threshhold:
            return 'Level 2'
        elif row['L3_average'] <= threshhold:
            return 'Level 3'
        elif row['L4_average'] <= threshhold:
            return 'Level 4'
        elif row['L5_average'] <= 100:
            return 'Level 5'
        else:
            return 'Level 6'

#     #create a new column in the combined_df by applying the function above for each provider
    combined_df['Recommendation'] = combined_df.apply(calculate_rec, axis=1)
    #return only the selected columns below
    combined_df = combined_df[['ProviderName', 'L1_average', 'L2_average', 'L3_average', 'L4_average', 'L5_average', 'Recommendation']]

    return combined_df
          

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
            f"The Service Tariff of {provider} has a variance of {variance_averages[f'L{i}_ave']}% from "
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
            f"The Service Tariff of {provider} has a variance of {variance_averages['L5_ave']}% "
            "on LEVEL 5 and is hereby recommended to TARIFF LEVEL 6."
        )
        return table_df, rec_level_6

    # Fallback (should rarely happen now)
    fallback_rec = (
        f"The Service Tariff of {provider} does not meet any of the thresholds for recommendation "
        "to a specific TARIFF LEVEL based on the current variance analysis."
    )
    return table_df, fallback_rec


    # # Default recommendation if no level matches criteria
    # rec_default = (
    #     f"The Service Tariff of {provider} has a variance that exceeds thresholds for "
    #     "TARIFF LEVELS 1–5 and is hereby recommended to BUPA LEVEL."
    # )
    # return table_df, rec_default



#apply the aggregate_provider_tariff function above to each category of providers and create a new column in the returned dataframe to indicate the provider category on TOSHFA
Level_1_providers_df = aggregate_provider_tariff('LEVEL 1', 'Level_1')
Level_1_providers_df['TOSHFA Level'] = 'Level 1'
Level_2_providers_df = aggregate_provider_tariff('LEVEL 2', 'Level_2')
Level_2_providers_df['TOSHFA Level'] = 'Level 2'
Level_3_providers_df = aggregate_provider_tariff('LEVEL 3', 'Level_3')
Level_3_providers_df['TOSHFA Level'] = 'Level 3'
Level_4_providers_df = aggregate_provider_tariff('LEVEL 4', 'Level_4')
Level_4_providers_df['TOSHFA Level'] = 'Level 4'
Level_5_providers_df = aggregate_provider_tariff('LEVEL 5', 'Level_5')
Level_5_providers_df['TOSHFA Level'] = 'Level 5'
Level_6_providers_df = aggregate_provider_tariff('LEVEL 6', 'Level_5')
Level_6_providers_df['TOSHFA Level'] = 'Level 6'

#combine all the providers in the different categories above to get a list with all the providers
all_providers_df = pd.concat([Level_1_providers_df,Level_2_providers_df,Level_3_providers_df,Level_4_providers_df,Level_5_providers_df,Level_6_providers_df], axis=0)
all_providers_df = all_providers_df[['ProviderName', 'TOSHFA Level', 'Recommendation']]
# optical_providers_df = aggregate_provider_tariff_gp('Dental', 'Level_1')
# optical_providers_df['TOSHFA Level'] = 'BASIC'
# st.write(all_providers_df[['Recommendation'] == 'Level 1'].reset_index(drop=True))
#this function adds a filter to the sidebar to enable us filter the final displayed categorisation by the model based on their recommended tariff level
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

# Define a function to apply custom styling
def highlight_columns(val, column_names, color):
    if val.name in column_names:
        return f'background-color: {color}'
    return ''

#add a radio button to select the task to be carried out
select_task = st.sidebar.radio('Select Task', options=['Check Provider CPT Mapping Compliance', 'Check Provider Classification'])

#below is the set of instructions to be executed when the first option is selected
if select_task == 'Check Provider CPT Mapping Compliance':
    st.title('CPTCode Mapping Compliance Tracker')
    #merge the provider tariff table with the standard cpt code and description master sheet and join on the CPTcode column
    provider_mapped_df = pd.merge(merged_provider_tariff, service_details, on=['CPTCode'], how='inner', indicator='Exist')
    #rename the CPTDescription column to ProviderDescription
    provider_mapped_df.rename(columns={'CPTDescription':'ProviderDescription'}, inplace=True)
    #change the provider description column to upper case to enhance comparison
    provider_mapped_df['ProviderDescription'] = provider_mapped_df['ProviderDescription'].str.upper()
    #create a list of unique providers and assign to a variable
    unq_provider = provider_mapped_df['ProviderName'].unique()
    #create a select box with the unique providers and assign the selected provider to a variable
    select_provider = st.sidebar.selectbox('Select Provider', options=unq_provider)
    #create a seperate dataframe for all services relating to a selected provider
    select_provider_df = provider_mapped_df[provider_mapped_df['ProviderName'] == select_provider]
    #create a new column using the fuzzy score function above to assign a score to each service based on their compatibility
    select_provider_df['Compatibility_Score'] = select_provider_df.apply(lambda row: compare_cpt_description(row['ProviderDescription'], row['StandardDescription']), axis=1)
    #select the columns to be displayed and reset the index
    select_provider_df = select_provider_df[['CPTCode', 'ProviderDescription', 'StandardDescription','Compatibility_Score', 'Amount']].reset_index(drop=True)
    #calculate the average compatibility score for the selected provider
    avg_compatibility_score = round(select_provider_df['Compatibility_Score'].mean(),2)
    #display a header for the title of the displayed table
    st.subheader(f'{select_provider} CPT Code Mapping with AVON Standard Service Description and TARIFF')
    #display the selected data to the portal
    st.write(select_provider_df)
    #add a download button to download the dataframe to excel
    st.download_button(
                label=f'Download {select_provider} tariff data as Excel File',
                data=select_provider_df.to_csv().encode('utf-8'),
                file_name=f'{select_provider} CPT Compatibility data.csv',
                mime='text/csv',
                )

    #display the average compatibility score to the portal
    st.subheader(f'The Average Compatibility Score of {select_provider} Service Description with AVON Standard Service Description is {avg_compatibility_score}%')

#set of instructions to be executed when the other task is selected.
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
    selected_provider_df = merged_provider_standard_tariff[merged_provider_standard_tariff['ProviderName'] == provider].reset_index(drop=True)
    selected_provider_df = selected_provider_df[['Category', 'CPTCode', 'ProviderServiceDesc', 'StandardServiceDesc', 'Match_Score', 'Amount',
                                                 'Level_1', 'Tariff-L1%', 'Level_2', 'Tariff-L2%', 'Level_3', 'Tariff-L3%', 'Level_4',
                                                 'Tariff-L4%', 'Level_5', 'Tariff-L5%']]
    #drop duplicates from selected_provider_df
    selected_provider_df = selected_provider_df.drop_duplicates(subset=['CPTCode', 'ProviderServiceDesc']).reset_index(drop=True)
    
    var_df, rec = calculate_rec(selected_provider_df, provider, location)
    
    st.subheader(f'Service Tariff Table for {provider}')
    st.dataframe(selected_provider_df)
    st.subheader(f'{provider} Service Tariff Variance from each Standard Tariff Level')
    st.dataframe(var_df)
    st.header('RECOMMENDATION')
    st.write(rec)

    return provider, selected_provider_df

if select_task == 'Check Provider Classification':
    # Sidebar to select provider class
    provider_class = st.sidebar.selectbox(
        label='Select Current Provider Class', 
        options=['ALL', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6']
    )
    
    # Filter the provider dataset based on the selected class
    if provider_class == 'ALL':
        provider_df = display_data(all_providers_df)
        unique_providers = merged_provider_standard_tariff['ProviderName'].unique()
    else:
        provider_df = display_data(eval(f"{provider_class.replace(' ', '_')}_providers_df"))
        unique_providers = merged_provider_standard_tariff.loc[
            merged_provider_standard_tariff['ProviderClass'] == provider_class.upper(), 
            'ProviderName'
        ].unique()
    
    # Display data and perform analysis for the selected class
    provider, selected_provider_df = display_provider_data(provider_class, provider_df, unique_providers)

    
    #add a download button to enable download of the selected provider tariff data
    st.download_button(
                label=f'Download {provider} tariff data as Excel File',
                data=selected_provider_df.to_csv().encode('utf-8'),
                file_name=f'{provider} tariff data.csv',
                mime='text/csv',
                )
    #add a download button to enable download of the selected provider class classification
    st.download_button(
                label=f'Download {provider_class} tariff Classification as Excel File',
                data=provider_df.to_csv().encode('utf-8'),
                file_name=f'{provider_class} Providers Tariff Classification.csv',
                mime='text/csv',
                )
    

# # Specify the columns and the color you want to highlight
# highlight_columns_names = ['Age', 'Score']
# highlight_color = 'yellow'

# # Apply the styling using the Styler object
# styled_df = df.style.apply(highlight_columns, column_names=highlight_columns_names, color=highlight_color, axis=None)