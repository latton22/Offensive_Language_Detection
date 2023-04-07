import torch
import numpy as np
import transformers
import torchvision
import IPython

from transformers import AutoTokenizer
from dataset import make_en_test_dataset, make_jp_test_dataset, make_labeledJP_dataset, make_japanese_dataset, DOLDataset
from model import Classifier
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from captum.attr import visualization as viz
from lime.lime_text import LimeTextExplainer

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

def test(test_loader, model, explainer, device):
    def predictor(texts):
        """LIMEに渡す用に、推論結果（確率）を計算する
        Args:
            texts: 文章のリスト
        Returns:
            推論結果（確率）のリスト
        """
        # 文章をID化する
        encoding = tokenizer.batch_encode_plus(
                    texts,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=config.max_token_len,
                    truncation = True)

        input_ids = torch.tensor(encoding['input_ids']).to(device)
        # print(input_ids.size())

        # 学習済みモデルによる推論
        with torch.no_grad():
            output = model(input_ids)

        # 推論結果をSoftmax関数を通して確率表現にする
        probas = F.softmax(output, dim=-1).cpu().detach().numpy()

        return probas

    outputs_list = []
    test_pred  = []
    test_label = []
    pred_prob  = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.nn.functional.one_hot(data['labels'].long(), num_classes=config.output_class_num) # Integer --> One-hot
            labels = labels.to(device, torch.float64)

            outputs = model(ids, mask)
            outputs_list += outputs.cpu().detach().numpy().copy().tolist()

            for _ids, _mask in zip(ids, mask):
                exp = explainer.explain_instance((_ids, _mask), model, num_features=6, num_samples=config.output_class_num)
                prob = exp.predict_proba
                pred = np.argmax(exp.predict_proba)
                attention = [0]*len(config.max_token_len)

                explanation = exp.as_map()[pred_id]
                for exp in explanation:
                    if(exp[1]>0):
                        attention[exp[0]]=exp[1]
                word_attributions.append(attention)

                test_pred.append(pred)
                pred_prob.append(prob)

            # 予測結果・正解ラベルの系列を生成
            test_label += torch.argmax(labels, dim=-1).cpu().detach().numpy().copy().tolist()  # バッチサイズの長さの正解ラベル配列

    pred_class = [['NOT', 'OFF'][idx] for idx in test_pred]
    true_class = [['NOT', 'OFF'][idx] for idx in test_label]
    attr_score = [sum(attrs) for attrs in word_attributions]

    return word_attributions, pred_prob, pred_class, true_class, attr_score

def store_feature(module, input, output):
    global feature
    feature = output

def print_model(module, name="model", depth=0):
    if len(list(module.named_children())) == 0:
        print(f"{' ' * depth} {name}: {module}")
    else:
        print(f"{' ' * depth} {name}: {type(module)}")

    for child_name, child_module in module.named_children():
        if isinstance(module, torch.nn.Sequential):
            child_name = f"{name}[{child_name}]"
        else:
            child_name = f"{name}.{child_name}"
        print_model(child_module, child_name, depth + 1)

def main():
    gpu_id = sys.argv[2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+ str(gpu_id) if use_cuda else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(config.language_model)
    explainer = LimeTextExplainer(class_names=["NOT", "OFF"], split_expression='\s+', random_state=333, bow=False)

    for i, seed in enumerate(config.random_seeds[:1]):
        out_dir = sys.argv[3] + '/seed_{}'.format(i+1)

        # setting model
        model = Classifier()
        params = torch.load(out_dir+config.test_model_path)
        model.load_state_dict(params)
        model = model.to(device)

        # print_model(model)

        """
        # for test data
        test_df = make_en_test_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir + config.test_result_path + '/OLID_Test'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # for test data
        test_df = make_jp_test_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/Noisy_JPdataset'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # Labeled data by fujihara
        test_df = make_labeledJP_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/labeled_by_fujihara'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # japanese offensive language dataset
        test_df = make_japanese_dataset()
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/JOLD'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.only_label_A_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/only_label_a'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.only_label_B_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/only_label_b'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.split_4_groups_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/split_4_groups'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.absolute_label_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/absolute_label'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []
        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)
        """

        # japanese offensive language dataset
        test_df = make_japanese_dataset(data_path=config.only_label_B_dataset)
        test_set    = DOLDataset(test_df, tokenizer)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        dir_path = out_dir+config.test_result_path + '/only_label_b'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=False)
        rationale_path = dir_path + '/rationale.html'
        word_attributions, pred_prob, pred_class, true_class, attr_score = test(test_loader, model, explainer, device)
        raw_input_ids = test_df.tweet.values.tolist()
        convergence_score = [0 for _ in range(len(raw_input_ids))]
        attr_class = ["NOT" for _ in range(len(raw_input_ids))]
        data_recode = []

        for i in range(len(raw_input_ids)):
            score_vis = viz.VisualizationDataRecord(
                            word_attributions = word_attributions[i],
                            pred_prob = pred_prob[i],
                            pred_class = pred_class[i],
                            true_class = true_class[i],
                            attr_class = attr_class[i],
                            attr_score = attr_score[i],
                            raw_input_ids = raw_input_ids[i],
                            convergence_score = convergence_score[i])
            data_recode.append(score_vis)
        html = viz.visualize_text(data_recode, True)
        with open(rationale_path, "w") as file:
            file.write(html.data)

# %% 実行部
if __name__ == '__main__':
    install()
    console = Console()
    try:
        main()
    except:
       console.print_exception()
