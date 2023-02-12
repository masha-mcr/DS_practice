import sys
import pandas as pd
import os

from sales_pred_filiankova.features.feature_engineering import FeatureGenerator


if __name__ == '__main__':
    source_dir = sys.argv[1]
    feat_gen = FeatureGenerator(dest_dir=sys.argv[2],
                                shops=pd.read_csv(os.path.join(source_dir, 'shops.csv')),
                                items=pd.read_csv(os.path.join(source_dir, 'items.csv')),
                                sales=pd.read_csv(os.path.join(source_dir, 'sales_train.csv'), parse_dates=['date'],
                                                  dayfirst=True),
                                item_categories=pd.read_csv(os.path.join(source_dir, 'item_categories.csv')))
    feat_gen.saturate_shops()
    feat_gen.saturate_items()
    feat_gen.saturate_categories()
    feat_gen.get_item_sales_lags()
    feat_gen.get_price_lags()
    feat_gen.get_shop_variety()
    feat_gen.get_item_spread()
    feat_gen.get_total_sales_lags()
