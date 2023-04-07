import torch
import numpy as np
import transformers

from transformers import AutoTokenizer
from dataset import make_en_test_dataset, make_jp_test_dataset, make_labeledJP_dataset, make_japanese_dataset, DOLDataset
from model import Classifier
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from rich.console import Console
from rich.traceback import install
from rich.progress import track

import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir)
config = import_module(config_bname)

def test(test_loader, model, criterion, device):
    outputs_list = []
    test_pred  = []
    test_label = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.nn.functional.one_hot(data['labels'].long(), num_classes=config.output_class_num) # Integer --> One-hot
            labels = labels.to(device, torch.float64)

            outputs = model(ids, mask)
            outputs_list += outputs.cpu().detach().numpy().copy().tolist()

            # Lossの算出
            loss = criterion(outputs, labels)
            test_loss = loss.item()
            del loss

            # 予測結果・正解ラベルの系列を生成
            test_pred  += torch.argmax(outputs, dim=-1).cpu().detach().numpy().copy().tolist() # バッチサイズの長さの予測ラベル配列
            test_label += torch.argmax(labels, dim=-1).cpu().detach().numpy().copy().tolist()  # バッチサイズの長さの正解ラベル配列
    return test_loss, test_pred, test_label, outputs_list


def evaluate_model(pred, label, score_path, confusion_matrix_path):
    print(pred)
    print(label)
    pre_off = precision_score(label, pred, pos_label=1)
    pre_not = precision_score(label, pred, pos_label=0)
    rec_off = recall_score(label, pred, pos_label=1)
    rec_not = recall_score(label, pred, pos_label=0)
    f1_off = f1_score(label, pred, pos_label=1)
    f1_not = f1_score(label, pred, pos_label=0)
    macro_f1 = f1_score(label, pred, average='macro')
    acc_off = accuracy_score(label, pred)
    with open(score_path, mode='a') as f:
        f.write('Precision_off, {}\n'.format(pre_off))
        f.write('Precision_not, {}\n'.format(pre_not))
        f.write('Recall_off, {}\n'.format(rec_off))
        f.write('Recall_not, {}\n'.format(rec_not))
        f.write('F1_off, {}\n'.format(f1_off))
        f.write('F1_not, {}\n'.format(f1_not))
        f.write('Macro F1, {}\n'.format(macro_f1))
        f.write('Accuracy, {}\n'.format(acc_off))

    cm = confusion_matrix(label, pred)
    print(cm)
    with open(confusion_matrix_path, mode='w') as f:
        f.write(',NOT,OFF\n')
        f.write('NOT,{},{}\n'.format(cm[0][0], cm[0][1]))
        f.write('OFF,{},{}\n'.format(cm[1][0], cm[1][1]))


def main():
    gpu_id = sys.argv[2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+ str(gpu_id) if use_cuda else "cpu")
    print(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    tokenizer = AutoTokenizer.from_pretrained(config.language_model)

    for i, seed in enumerate(config.random_seeds):
        out_dir = sys.argv[3] + '/seed_{}'.format(i+1)

        # setting model
        model = Classifier()
        params = torch.load(out_dir+config.test_model_path)
        model.load_state_dict(params)
        model = model.to(device)

        # for test data
        test_df = make_en_test_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir + config.test_result_path + '/OLID_Test'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # for test data
        test_df = make_jp_test_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/Noisy_JPdataset'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # Labeled data by fujihara
        test_df = make_labeledJP_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/labeled_by_fujihara'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # japanese offensive language dataset
        test_df = make_japanese_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/JOLD'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.only_label_A_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/only_label_a'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.only_label_B_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/only_label_b'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.split_4_groups_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/split_4_groups'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.absolute_label_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/absolute_label'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        score_path = dir_path + '/score.csv'
        confusion_matrix_path = dir_path + '/confusion_matrix.csv'

        test_loss, test_pred, test_label, outputs_list = test(test_loader, model, criterion, device)
        with open(score_path, mode='w') as f:
            f.write('MSE Loss, {}\n'.format(test_loss))
        evaluate_model(test_pred, test_label, score_path, confusion_matrix_path)

        predicted_label_path = dir_path + '/predicted_label.tsv'
        test_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        test_df['NOT'] = outputs_list[:,0]
        test_df['OFF'] = outputs_list[:,1]
        test_df.to_csv(predicted_label_path, sep='\t', index=False)

# %% 実行部
if __name__ == '__main__':
    install()
    console = Console()
    try:
        main()
    except:
       console.print_exception()
