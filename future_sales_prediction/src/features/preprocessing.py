import os.path
import pandas as pd
import sys

from sales_pred_filiankova.features import Preprocessor


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(sys.argv[1], 'train.csv'))
    val = pd.read_csv(os.path.join(sys.argv[1], 'val.csv'))

    preprocessor = Preprocessor(target_col='item_cnt_month', no_log_target=True, outlier_prob=0.03,
                                min_samples_leaf=1, smoothing=1.0)

    train_x, train_y, val_x, val_y = preprocessor.preprocess_data(train_data=train, val_data=val)

    train_x.to_csv(os.path.join(sys.argv[2], f'train_x.csv'), index=False)
    train_y.to_csv(os.path.join(sys.argv[2], f'train_y.csv'), index=False)
    val_x.to_csv(os.path.join(sys.argv[2], f'val_x.csv'), index=False)
    val_y.to_csv(os.path.join(sys.argv[2], f'val_y.csv'), index=False)
