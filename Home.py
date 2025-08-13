import streamlit as st
import pandas as pd
import pyodbc
import datetime as dt
from PIL import Image
import os

st.set_page_config(page_title='Tariff Review Portal', layout='wide', initial_sidebar_state='expanded')

# Add image header
image = Image.open('tariff_portal.png')
st.image(image, use_column_width=True)

# Database credentials from environment variables
server = os.environ.get('server_name')
database = os.environ.get('db_name')
username = os.environ.get('db_username')
password = os.environ.get('db_password')

# Database connection
try:
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
except pyodbc.Error as ex:
    st.error(f"Database connection failed: {ex}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
choice = st.sidebar.segmented_control(
    "Select Module",
    ["Service Tariff Categorization", "Drug Tariff Categorization", "TOSHFA Provider Tariff", "Referral Module"]
)

# Function to execute a module in a separate namespace
def execute_module(module_name):
    try:
        with open(module_name, encoding = 'utf-8') as file:
            module_code = file.read()
        module_namespace = {'conn': conn, 'st': st, 'pd': pd, 'dt': dt}
        exec(module_code, module_namespace)
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error(s) occurred: {e}")

# Load selected module
if choice == "Service Tariff Categorization":
    execute_module("Service Tariff Categorization.py")
elif choice == "Drug Tariff Categorization":
    execute_module("Drug Tariff Categorization.py")
elif choice == "TOSHFA Provider Tariff":
    execute_module("Provider Categorization Module.py")
elif choice == "Referral Module":
    execute_module("Referral Module.py")
