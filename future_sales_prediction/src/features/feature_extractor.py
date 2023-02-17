import pandas as pd
import os
import sys
from datetime import date
import json
from sales_pred_filiankova.features import FeatureExtractor, ValidationSchema

if __name__ == '__main__':
    lookup_datasets = {'shops': pd.read_csv(os.path.join(sys.argv[2], 'shops.csv'),
                                            parse_dates=['first_shop_appearance'], dayfirst=True),
                       'items': pd.read_csv(os.path.join(sys.argv[2], 'items.csv'),
                                            parse_dates=['first_item_appearance'], dayfirst=True),
                       'item_categories': pd.read_csv(os.path.join(sys.argv[2], 'item_categories.csv'))}

    lag_datasets = {'shop_variety': pd.read_csv(os.path.join(sys.argv[2], 'shop_variety_monthly.csv')),
                    'item_spread': pd.read_csv(os.path.join(sys.argv[2], 'item_spread_monthly.csv')),
                    'item_price': pd.read_csv(os.path.join(sys.argv[2], 'item_price_monthly.csv')),
                    'item_sales': pd.read_csv(os.path.join(sys.argv[2], 'item_sales_monthly.csv')),
                    'total_sales': pd.read_csv(os.path.join(sys.argv[2], 'total_sales_monthly.csv')),
                    }

    extractor = FeatureExtractor(feature_datasets=lookup_datasets, lag_datasets=lag_datasets)

    sales = pd.read_csv(sys.argv[1], parse_dates=['date'], dayfirst=True)

    split_config_f = open('features/split_config.json')
    split_config = json.load(split_config_f)
    split_config_f.close()

    vs = ValidationSchema(data=sales)
    train_data, val_data = vs.split_data(
        train_start=date(split_config['train_start_y'], split_config['train_start_m'], 1),
        val_start=date(split_config['val_y'], split_config['val_m'], 1))

    train, val = extractor.extract_features(train_data=train_data, val_data=val_data)

    if not os.path.exists(sys.argv[3]):
        os.mkdir(sys.argv[3])

    val.to_csv(os.path.join(sys.argv[3], f'val.csv'), index=False)
    train.to_csv(os.path.join(sys.argv[3], f'train.csv'), index=False)
