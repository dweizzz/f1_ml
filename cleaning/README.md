FULL_F1_DF.csv: all historic data before 2023 season 
final_cleaned.csv: prepared data for analysis, made direction columns binary
final_df213.csv: cleaned result.csv
more_cols_less_rows.csv: part of data cleaning, removed irrelevant/null data [old]
race1.csv: prepared dataset for first race of 2023 season; inputted current data for each driver, weather, track, etc.
result.csv: from erghast api, with missing data added
all_data_after_race1.csv: all data up until end of race 1

cleaning.ipynb: added columns to result.csv
cleaning2.ipynb: created more_cols_less_rows
cleaning3.ipynb: created FULL_F1_DF
race1_update.ipynb: append race1 data to FULL_F1_DF
