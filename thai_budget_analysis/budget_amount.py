import pandas as pd
import re
import os
import logging
import json
import unicodedata
import plotly.express as px

# Configure logging to provide info and error messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BudgetCal:
    def __init__(self, config):
        """
        Initialize the BudgetCal class with the provided configuration.

        Args:
            config (dict): Configuration parameters loaded from 'config.json'.
        """
        self.config = config
        logging.info("Configuration loaded successfully.")

    def read_csv_files(self, dir_path):
        """
        Read CSV files from the specified directory path.

        Args:
            dir_path (str): The directory path to read CSV files from.

        Returns:
            generator: Generator yielding tuples of (file_name, dataframe).
        """
        csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
        if not csv_files:
            logging.warning("No CSV files found in the directory.")
            return None
        return ((f, pd.read_csv(os.path.join(dir_path, f), encoding='utf-8')) for f in csv_files)

    def merge_dataframes(self, file_generator, pattern):
        """
        Merge dataframes generated from CSV files based on the provided pattern.

        Args:
            file_generator (generator): Generator yielding tuples of (file_name, dataframe).
            pattern (str): Regular expression pattern to extract year from file names.

        Returns:
            pd.DataFrame: Merged dataframe or None if no dataframes are generated.
        """
        if file_generator is None:
            return None
        dfs = []
        for file_name, df in file_generator:
            match = re.search(pattern, file_name)
            if match:
                year = int(match.group(1))
                df['BUDGET_YEAR'] = year
                dfs.append(df)
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            logging.info("CSV files merged successfully.")
            return merged_df
        else:
            logging.warning("No valid dataframes were generated.")
            return None

    def prepare_data(self, df, group_by):
        """
        Prepare data for analysis by converting columns to appropriate data types and applying filters.

        Args:
            df (pd.DataFrame): Input dataframe.
            group_by (str): Column name to group data by.

        Returns:
            pd.DataFrame: Prepared and aggregated dataframe.
        """
        if group_by not in ['MINISTRY', 'BUDGETARY_UNIT']:
            logging.error("Invalid group_by parameter. Please provide 'MINISTRY' or 'BUDGETARY_UNIT'.")
            return None
        if 'AMOUNT' not in df.columns or 'OBLIGED?' not in df.columns:
            logging.error("Required column missing in the dataframe.")
            return None

        # Ensure column formats are correct
        df['OBLIGED?'] = df['OBLIGED?'].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
        df.dropna(subset=['OBLIGED?'], inplace=True)
        df['AMOUNT'] = df['AMOUNT'].str.replace(',', '').astype(float) / 1e6
        df['FISCAL_YEAR'] = df['BUDGET_YEAR'].astype(int) # Modified this line to use 'BUDGET_YEAR' instead of 'FISCAL_YEAR'

        obliged_year = self.config.get('fy')
        # Filter the data based on the 'OBLIGED?' column and the specified year (if provided)
        if obliged_year:
            df = df[((df['OBLIGED?'] == True) & (df['FISCAL_YEAR'] == obliged_year)) | (df['OBLIGED?'] == False)]

        return df.groupby([group_by, 'BUDGET_YEAR'])['AMOUNT'].sum().reset_index()

    def plot_data(self, df, title, direction='top', n_entries=5):
        """
        Plot data using Plotly Express. This function creates a bar chart based on the DataFrame provided.
        It allows for customization of the plot appearance and data representation through various parameters.
        
        Args:
            df (pd.DataFrame)   : The DataFrame containing the data to be plotted. 
                                It must contain columns specified in 'group_by' and 'AMOUNT'.
            title (str)         : The title of the bar chart.
            direction (str)     : Determines the sorting direction of data points in the plot; 
                                'top' for descending order to show the top entries, 'bottom' for ascending.
            n_entries (int)     : Specifies the number of top entries to display in the bar chart.

        Notes:
            - The 'group_by' configuration must be a column in the DataFrame, as it is used to color-code the bars.
            - If the DataFrame is empty, contains none of the required columns, or is not passed, 
            the function logs an error and exits without creating a plot.
        """
        # Check if DataFrame is valid and contains the necessary 'group_by' column for plotting.
        if df is None or self.config['group_by'] not in df.columns:
            logging.error(f"Dataframe is either None or missing required column for plotting: {self.config['group_by']}")
            return
        
        # Sort and limit the data based on 'AMOUNT' according to the specified direction and number of entries.
        if direction == 'top':
            df = df.sort_values(by='AMOUNT', ascending=False)
        else:
            df = df.sort_values(by='AMOUNT', ascending=True)
        df = df.groupby('BUDGET_YEAR').head(n_entries)  # Group by 'BUDGET_YEAR' and take the top 'n_entries' per group.

        # Create the bar chart using Plotly Express.
        fig = px.bar(df, x='BUDGET_YEAR', y='AMOUNT', color='BUDGETARY_UNIT', title=title)
        
        # Display the plot.
        fig.show()


if __name__ == '__main__':
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        bg_cal = BudgetCal(config)
        file_generator = bg_cal.read_csv_files(config['dir_path'])
        file_pattern = config.get("file_pattern", r"(\d+)_Budget_red_stripe.csv")
        merged_df = bg_cal.merge_dataframes(file_generator, file_pattern)
        if merged_df is not None:
            plot_prepared_df = bg_cal.prepare_data(merged_df, config['group_by'])
            bg_cal.plot_data(plot_prepared_df, config['plot']['title'], config['plot']['labels'], config['plot']['direction'], config['plot']['n_entries'])
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
