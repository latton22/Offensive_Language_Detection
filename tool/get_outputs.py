import torch
import numpy as np
import transformers

from transformers import AutoTokenizer
from dataset import UnlabelDataset, make_unlabel_dataset
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

def test(test_loader, model, device):
    outputs_list = []
    test_pred = []
    model.eval()
    with torch.no_grad():
        for data in track(test_loader, total=len(test_loader), description='Processing...'):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.nn.functional.one_hot(data['labels'].long(), num_classes=config.output_class_num) # Integer --> One-hot
            labels = labels.to(device, torch.float64)

            outputs = model(ids, mask)
            outputs_list += outputs.cpu().detach().numpy().copy().tolist()

            # 予測結果・正解ラベルの系列を生成
            test_pred  += torch.argmax(outputs, dim=-1).cpu().detach().numpy().copy().tolist()
    return test_pred, outputs_list

def main():
    tokenizer = AutoTokenizer.from_pretrained(config.language_model)

    gpu_id = sys.argv[2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+ str(gpu_id) if use_cuda else "cpu")
    print(device)

    # setting model
    model = Classifier()
    params = torch.load(config.test_model_path)
    model.load_state_dict(params)
    model = model.to(device)

    # for test data
    data_path_list = ['/home/usrs/tomoki.fujihara.p3/JPOLID/compare_text_source/code/out/tweet2020-08-18.tsv', '/home/usrs/tomoki.fujihara.p3/JPOLID/compare_text_source/code/out/tweet2020-08-19.tsv']
    for data_path in data_path_list:
        id_df, text_df = make_unlabel_dataset(data_path)
        test_set   = UnlabelDataset(text_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2)
        dir_path = '../out/model_outputs/'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)

        test_pred, outputs_list = test(test_loader, model, device)

        predicted_label_path = dir_path + os.path.basename(data_path)
        id_df['text'] = text_df['text']
        id_df['predicted'] = [['NOT', 'OFF'][idx] for idx in test_pred]
        outputs_list = np.array(outputs_list)
        id_df['NOT'] = outputs_list[:,0]
        id_df['OFF'] = outputs_list[:,1]
        id_df.to_csv(predicted_label_path, sep='\t', index=False)

# %% 実行部
if __name__ == '__main__':
    install()
    console = Console()
    try:
        main()
    except:
       console.print_exception()
