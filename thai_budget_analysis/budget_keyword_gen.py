from budget_scrap import BudgetSca
from budget_amount import BudgetCal
import pandas as pd
import tqdm 
from openpyxl.workbook import Workbook
from tqdm import tqdm
import json
import logging

def save_df_to_excel_with_progress(df, filename, chunk_size=2000):
    # Initialize Excel writer
    writer = pd.ExcelWriter(filename, engine='openpyxl', mode='w')
    
    # Create a sheet and make it visible
    writer.book.create_sheet('Sheet1')
    writer.book.active = writer.book['Sheet1']

    # Number of chunks
    num_chunks = (len(df) - 1) // chunk_size + 1

    try:
        # Iterate over chunks
        for i in tqdm(range(num_chunks), desc='Exporting DataFrame to Excel'):
            start_row = i * chunk_size
            end_row = start_row + chunk_size
            
            # Write chunk to Excel
            df[start_row:end_row].to_excel(writer, startrow=start_row, index=False, sheet_name='Sheet1', header=False if i > 0 else True)

        logging.info("Data exported saving . . .")       
        writer.close()
        logging.info("Data exported to Excel successfully.")
    except Exception as e:
        logging.error(f"Error exporting data to Excel: {e}")
        writer.close()

    return

if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    bg_cal = BudgetCal(config)
    bg_sca = BudgetSca(config, engine = config.get("TH_NLP_engine"))
    file_generator = bg_cal.read_csv_files(config['dir_path'])
    file_pattern = config.get("file_pattern", r"(\d+)_Budget_red_stripe.csv")
    group_by = config.get("group_by")
    merged_df = bg_cal.merge_dataframes(file_generator, file_pattern)
    if merged_df is not None:
        scrap_prepared_df = bg_sca.keyword_get(merged_df)
        transformed_df = bg_sca.df_transform(scrap_prepared_df, group_by, config['fy'])
        save_df_to_excel_with_progress(transformed_df, '.\\Output\\transformed_df.xlsx')
        logging.info("Data transformation completed successfully.")