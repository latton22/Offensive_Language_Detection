import pandas as pd
import torch
import emoji
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from wordsegment import load, clean, segment

import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir)
config = import_module(config_bname)

def make_dataset(seed=10):
    ##### データの読込
    # OLID
    # train_EN_df = pd.read_csv(config.train_data_EN_path, sep='\t', usecols=[0, 1, 2]) # 動作確認用の引数 : , nrows=10
    train_transJP_df = pd.read_csv(config.train_data_transJP_path, sep='\t', usecols=[0, 1, 2])
    # SOLID
    # train_AR_df = pd.read_csv(config.train_data_AR_path, sep='\t', usecols=[0, 1, 2])
    # train_DA_df = pd.read_csv(config.train_data_DA_path, sep='\t', usecols=[0, 1, 2])
    # train_TR_df = pd.read_csv(config.train_data_TR_path, sep='\t', usecols=[0, 1, 2])

    # データの分割 (random_stateの値を変えると, 分割結果が変わる)
    # train_EN, valid_EN = train_test_split(train_EN_df, test_size=0.1, shuffle=True, random_state=seed, stratify=train_EN_df['subtask_a'])
    train_transJP, valid_transJP = train_test_split(train_transJP_df, test_size=0.1, shuffle=True, random_state=seed, stratify=train_transJP_df['subtask_a'])
    # train = pd.concat([train_EN, train_transJP, train_AR_df, train_DA_df, train_TR_df], axis=0)
    # valid = pd.concat([valid_EN, valid_transJP], axis=0)
    train = train_transJP
    valid = valid_transJP
    train.dropna(subset=['tweet'], inplace=True)
    valid.dropna(subset=['tweet'], inplace=True)
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)

    return train, valid

def make_en_test_dataset(test_data_path=config.test_data_path):
    test = pd.read_csv(test_data_path, sep='\t', usecols=[0, 1, 2])
    return test

def make_jp_test_dataset(off_lang_path=config.test_unlabeled_offjp_path, not_lang_path=config.test_unlabeled_notoffjp_path):
    # データの読込
    test_offtext_df = pd.read_csv(off_lang_path.replace('tool_name', '日本語データ'), sep='\t')
    test_offtext_df = test_offtext_df.dropna(how='any', axis=0)
    test_offtext_df['subtask_a'] = 'OFF'
    test_nottext_df = pd.read_csv(not_lang_path.replace('tool_name', '日本語データ'), sep='\t')
    test_nottext_df = test_nottext_df.dropna(how='any', axis=0)
    test_nottext_df['subtask_a'] = 'NOT'
    # 確認
    test = pd.concat([test_offtext_df, test_nottext_df[:2500]], axis=0)
    # test['text'] = '@USER ' + test['text'] # '@USER 'をテキストの先頭に付与して学習データのテキストに近づける
    test = test.rename(columns={'text': 'tweet'})
    # print(test.head())

    return test

def make_labeledJP_dataset(data_path=config.labeled_by_fujihara):
    # データの読込
    test = pd.read_csv(data_path, sep=',')
    test = test.rename(columns={'text': 'tweet'})
    test.reset_index(drop=True, inplace=True)
    return test

def make_japanese_dataset(data_path=config.japanese_offensive_language_dataset):
    # データの読込
    test = pd.read_csv(data_path, sep='\t', encoding='shift_jis')
    test = test.rename(columns={'text': 'tweet'})
    test.reset_index(drop=True, inplace=True)
    return test

class DOLDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_token_len = config.max_token_len

        print("The shape of the data: {}".format(self.dataset.shape))
        self.texts, self.labels = self.get_datasets()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_token_len,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.Tensor(self.labels[idx])
        }

    def get_datasets(self):
        # テキストの前処理
        load()
        texts = []
        for tweet in self.dataset['tweet']:
            texts.append(tweet)

        # ラベルの前処理
        labels = torch.tensor([1. if l == 'OFF' else 0. for l in self.dataset['subtask_a']]) # str --> Integer
        return texts, labels
