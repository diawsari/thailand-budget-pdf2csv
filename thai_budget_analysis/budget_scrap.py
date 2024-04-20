import pandas as pd
import re
import os
import time
import logging
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
import unicodedata
from collections import Counter
from progress.bar import Bar
from concurrent.futures import ThreadPoolExecutor
import threading
import ast
import numpy as np
import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BudgetSca:
    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        logging.info("Configuration loaded successfully.")

    def clean_bu(self, bu):
        """Normalize and clean budget unit text.

        Args:
            bu (str): The budget unit text to be cleaned.

        Returns:
            str or None: The cleaned budget unit text, or None if the input is NaN.

        """
        if pd.isna(bu):
            return None
        elif isinstance(bu, str):
            bu = unicodedata.normalize('NFKC', bu)
            bu = re.sub(r'\s+', ' ', bu).strip()
        return bu

    def tokenize_words(self, text):
        """
        Tokenize the given text by removing whitespace and stopwords, and keeping significant words only.

        Parameters:
            text (str): The text to be tokenized.

        Returns:
            list: A list of significant words after tokenization.
        """
        words = pythainlp.word_tokenize(text, engine=self.engine, keep_whitespace=False)
        stopwords = pythainlp.corpus.thai_stopwords()
        words = word_tokenize(text, keep_whitespace=False)
        result = [word for word in words if word not in stopwords and len(word) > 2]
        return result

    def process_field(self, field):
        """
        Process each field: split, clean, tokenize, and retrieve common words.

        Args:
            field (str): The field to be processed.

        Returns:
            str: The result of processing the field.

        """
        if pd.isna(field) or field.strip() == "":
            return ""
        entries = re.split(r',', field)
        cleaned_entries = filter(None, (self.clean_bu(entry.strip()) for entry in entries))
        all_words = sum((self.tokenize_words(entry) for entry in cleaned_entries), [])
        result = ','.join(word for word, _ in Counter(all_words).most_common(5))
        return result

    def parse_word_counts(self, word_counts):
        """Parse a string of word counts into a list of sub-words.

        Args:
            word_counts (str): A string containing word counts separated by commas.

        Returns:
            list: A list of sub-words extracted from the word_counts string.

        """
        if pd.isna(word_counts) or not word_counts.strip():
            return 
        word = self.process_field(word_counts)
        sub_word_list = re.split(r',', word)
        return sub_word_list

    def apply_parse_word_counts(self, df, column_name, bar):
        """
        Applies the parse_word_counts method to a specific column and updates the progress bar.

        Args:
            df (pandas.DataFrame): The DataFrame containing the column to be processed.
            column_name (str): The name of the column to be processed.
            bar (ProgressBar): The progress bar to be updated.

        Returns:
            tuple: A tuple containing the processed result and the column name.

        """
        logging.info(f"{threading.current_thread().name} starting processing on {column_name}")
        start_time = time.time()
        result = df[column_name].apply(self.parse_word_counts)
        bar.next()
        logging.info(f"{threading.current_thread().name} has finished processing {column_name} in {time.time() - start_time:.2f} seconds")
        return result, column_name

    def threaded_word_counts(self, df, columns):
        """
        Uses threading to apply word count parsing to multiple dataframe columns with progress monitoring, appending results as new columns.

        Args:
            df (pandas.DataFrame): The input dataframe.
            columns (list): A list of column names in the dataframe to apply word count parsing to.

        Returns:
            pandas.DataFrame: The modified dataframe with new columns containing the word count results.
        """
        bar = Bar('Processing Columns', max=len(columns))  # Initialize progress bar
        max_workers = os.cpu_count()  # Dynamically set the number of threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary mapping original column names to new column names where results will be stored
            future_to_column = {
                executor.submit(self.apply_parse_word_counts, df, col, bar): col for col in columns
            }
            for future in future_to_column:
                result, col_name = future.result()
                new_column_name = f"{col_name}_processed"  # Define new column name
                df[new_column_name] = result  # Append results to new column instead of replacing

        bar.finish()  # Clean up after the bar
        return df
    
    def keyword_get(self, df):
        """
        Process the given DataFrame to clean and transform the data.

        Args:
            df (pandas.DataFrame): The input DataFrame containing budget data.

        Returns:
            pandas.DataFrame: The processed DataFrame with cleaned and transformed data.
        """
        start_time = time.time()
        df['BUDGET_YEAR'] = df['BUDGET_YEAR'].astype(int)
        columns = ['PROJECT', 'OUTPUT', 'ITEM_DESCRIPTION', 'CATEGORY_LV1', 'CATEGORY_LV2', 'CATEGORY_LV3', 'CATEGORY_LV4', 'CATEGORY_LV5', 'CATEGORY_LV6'] # Define columns to be tokenized
        df = self.threaded_word_counts(df, columns)
        df.fillna(int(0), inplace=True) # Fill NaN values with 0

        # grouped.dropna(subset=columns, how='all', inplace=True)
        logging.info(f"keyword_get executed in {time.time() - start_time:.2f} seconds")
        return df

    def df_transform(self, df, groupby, cfg_fy):
        """
        Transforms the dataframe by grouping and aggregating data based on the MINISTRY/BUDGETARY_UNIT columns.

        Parameters:
            df (pandas.DataFrame): The input dataframe to be transformed. normally the output of keyword_get.
            groupby (str): The column name to group the data by. the MINISTRY/BUDGETARY_UNIT column.
            cfg_fy (int): The fiscal year to filter the data. current fiscal year to filter out obliged.

        Returns:
            pandas.DataFrame: The transformed dataframe with grouped and aggregated data.
        """
        def safe_literal_eval(data):
            """Safely evaluates strings that start with typical list, dict, or tuple indicators."""
            if isinstance(data, str) and data.startswith(('[', '{', '(')):
                try:
                    return ast.literal_eval(data)
                except (ValueError, SyntaxError) as e:
                    print(f"Error evaluating data: '{data}' with exception: {e}")
                    return data
            else:
                return data

        grouped_result = pd.DataFrame()  # DataFrame to hold all grouped results

        # Define the columns to include in the group by operation
        sca_cross_col = ['Word', 'REF_DOC' ,'REF_PAGE_NO', groupby, 'BUDGET_YEAR', 'FISCAL_YEAR', 'OBLIGED?', 'source_column', 'source_str']
        
        # Columns that are processed
        processed_columns = [
            'PROJECT_processed', 'OUTPUT_processed', 'ITEM_DESCRIPTION_processed', 
            'CATEGORY_LV1_processed', 'CATEGORY_LV2_processed', 'CATEGORY_LV3_processed', 
            'CATEGORY_LV4_processed', 'CATEGORY_LV5_processed', 'CATEGORY_LV6_processed'
        ]
        # Mapping of processed columns to original columns for source_str NLP cross-check
        column_mapping = {
            'PROJECT_processed': 'PROJECT',
            'OUTPUT_processed': 'OUTPUT',
            'ITEM_DESCRIPTION_processed': 'ITEM_DESCRIPTION',
            'CATEGORY_LV1_processed': 'CATEGORY_LV1',
            'CATEGORY_LV2_processed': 'CATEGORY_LV2',
            'CATEGORY_LV3_processed': 'CATEGORY_LV3',
            'CATEGORY_LV4_processed': 'CATEGORY_LV4',
            'CATEGORY_LV5_processed': 'CATEGORY_LV5',
            'CATEGORY_LV6_processed': 'CATEGORY_LV6'
        }

        # Preprocessing steps
        df['AMOUNT'] = df['AMOUNT'].astype(str)
        df['AMOUNT'] = df['AMOUNT'].str.replace(',', '').astype(float) / 1e6
        df['OBLIGED?'] = df['OBLIGED?'].astype(bool)
        df['FISCAL_YEAR'] = df['FISCAL_YEAR'].astype(int)
        df['REF_PAGE_NO'] = df['REF_PAGE_NO'].astype(str)
        df['source_column'] = None
        df['source_str'] = None
        
        # Explode the processed columns
        for col in tqdm(processed_columns, desc="Processing columns"):
            df[col] = df[col].apply(safe_literal_eval)
            df_exploded = df.explode(col)

            df_exploded['source_column'] = col
            df_exploded['Word'] = df_exploded[col].astype(str)
            if col in column_mapping:
                df_exploded['source_str'] = np.where(df_exploded[col] != 0, df_exploded[column_mapping[col]], np.nan)
            else:
                df_exploded['source_str'] = np.nan

            # Filter and group operations
            grouped = df_exploded.sort_values(by=sca_cross_col).groupby(sca_cross_col).agg(
                frequency=('AMOUNT', 'size'),
                total_amount=('AMOUNT', 'sum')
            ).reset_index()

            grouped_result = pd.concat([grouped_result, grouped], ignore_index=True) # Concatenate the grouped results each processed column

        # Filter the grouped results
        grouped_result.query("((`OBLIGED?` == True) & (FISCAL_YEAR == @cfg_fy)) | (`OBLIGED?` == False)", inplace=True)
        grouped_result.query("Word != '0'", inplace=True)

        return grouped_result


        