import F1_models
import pandas as pd


def get_kfcv_results():
    """
    Get df containing results from running cv with years as folds
    """

    df = {"Test Year": [], "Linear Precision": [],
          "SVM Precision": [], "Forest Precision": []}

    for test_year in range(1983, 2023):
        print(test_year)
        f1 = F1_models.F1_regressions(
            data_file="../cleaning/FULL_F1_DF.csv", test_year=test_year)
        f1.run_all_models(test_year=test_year)
        df['Test Year'].append(f1.test_year)
        df['Linear Precision'].append(f1.linear_score)
        df['SVM Precision'].append(f1.svm_score)
        df['Forest Precision'].append(f1.forest_score)
    df = pd.DataFrame(df)
    df.to_csv("kfcv_restults_2.csv")
    return df
