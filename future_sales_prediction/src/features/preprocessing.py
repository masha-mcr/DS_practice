import os.path
import pandas as pd
import sys

from sales_pred_filiankova.features.validation_schema import ValidationSchema
from sales_pred_filiankova.features.preprocessing import preprocess_data


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(sys.argv[1], 'train.csv'))
    val = pd.read_csv(os.path.join(sys.argv[1], 'val.csv'))

    train_x, train_y, val_x, val_y = ValidationSchema.x_y_split(train, val, 'item_cnt_month')
    train_x, train_y, val_x, val_y = preprocess_data(train_x, train_y, val_x, val_y)

    train_x.to_csv(os.path.join(sys.argv[2], f'train_x.csv'), index=False)
    train_y.to_csv(os.path.join(sys.argv[2], f'train_y.csv'), index=False)
    val_x.to_csv(os.path.join(sys.argv[2], f'val_x.csv'), index=False)
    val_y.to_csv(os.path.join(sys.argv[2], f'val_y.csv'), index=False)
