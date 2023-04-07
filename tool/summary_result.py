import pandas as pd
import numpy as np

from rich.console import Console
from rich.traceback import install
from rich.progress import track

import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir) # 相対パスで指定したconfigファイルを読み込めるように, sys.pathにconfigファイルの親ディレクトリを追加. 若干非推奨らしい.
config = import_module(config_bname) # configファイル名のモジュールを読み込み.

def main():
    dir_list = ['/OLID_Test', '/Noisy_JPdataset', '/labeled_by_fujihara', '/JOLD', '/only_label_a', '/only_label_b', '/split_4_groups', '/absolute_label']

    for dir in dir_list:
        index = ['MSE Loss', 'Precision_off', 'Precision_not', 'Recall_off', 'Recall_not', 'F1_off', 'F1_not', 'Macro F1', 'Accuracy']
        columns = ['seed_{}'.format(i+1) for i in range(len(config.random_seeds))] + ['average']
        df = pd.DataFrame(columns=columns, index=index)
        for i in range(len(config.random_seeds)):
            result_path = sys.argv[2]+'/seed_{}'.format(i+1) + config.test_result_path + dir + '/score.csv'
            tmp_df = pd.read_csv(result_path, sep=',', index_col=0, header=None)
            df['seed_{}'.format(i+1)] = tmp_df

        for idx in index:
            df.at[idx, 'average'] = np.mean(df.loc[idx, columns])
        print(df)
        df.to_csv(sys.argv[2] + '/summary' + dir + '.csv', header=True, index=True)


# %% 実行部
if __name__ == '__main__':
    install()
    console = Console()
    try:
        main()
    except:
       console.print_exception()
