import os.path
import pandas as pd
import sys
from sales_pred_filiankova.data.DQC import run_quality_check

if __name__ == '__main__':
    data_path = sys.argv[1]
    run_quality_check(sales=pd.read_csv(os.path.join(data_path, 'sales_train.csv'), parse_dates=['date'], dayfirst=True),
                      items=pd.read_csv(os.path.join(data_path, 'items.csv')),
                      shops=pd.read_csv(os.path.join(data_path, 'shops.csv')),
                      item_categories=pd.read_csv(os.path.join(data_path, 'item_categories.csv')),
                      test=pd.read_csv(os.path.join(data_path, 'test.csv')))






