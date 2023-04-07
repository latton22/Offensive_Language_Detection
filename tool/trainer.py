import torch
import numpy as np
import transformers

from transformers import AutoTokenizer
from dataset import DOLDataset, make_dataset
from model import Classifier
from transformers import get_linear_schedule_with_warmup
from sift import hook_sift_layer, AdversarialLearner

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

def fix_seed(seed):
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def updater(train_loader, valid_loader, model, optimizer, scheduler, criterion, adv, device):
    def logits_fn(model, *wargs, **kwargs):
        logits = model(kwargs['ids'], kwargs['mask'])
        return logits

    train_loss = []
    valid_loss = []
    valid_acc  = []
    model.train()

    for i, data in track(enumerate(train_loader), total=len(train_loader), description='Training model...'):
        optimizer.zero_grad() # 勾配の初期化

        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        labels = torch.nn.functional.one_hot(data['labels'].long(), num_classes=config.output_class_num) # Integer --> One-hot
        labels = labels.to(device, torch.float32)

        outputs = model(ids, mask)
        loss = criterion(outputs, labels) + adv.loss(outputs, logits_fn, loss_fn='mse', **{'ids': ids, 'mask': mask})
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        del loss
        print('train loss: {}'.format(train_loss[-1]))
    val_loss, val_acc = validation(valid_loader, model, criterion, device)
    valid_loss.append(val_loss)
    valid_acc.append(val_acc)
    print('train loss: {}, valid_loss: {}, valid_acc: {}'.format(train_loss[-1], valid_loss[-1], valid_acc[-1]))
    return train_loss, valid_loss, valid_acc

def validation(valid_loader, model, criterion, device):
    loss_sum = 0.0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.nn.functional.one_hot(data['labels'].long(), num_classes=config.output_class_num) # Integer --> One-hot
            labels = labels.to(device, torch.float64)

            outputs = model(ids, mask)

            # Lossの算出
            loss = criterion(outputs, labels)
            total += outputs.size(0)
            loss_sum += loss.item() * outputs.size(0)
            del loss

            # 正解したサンプル数のカウント
            pred = torch.argmax(outputs, dim=-1).cpu().numpy() # バッチサイズの長さの予測ラベル配列
            labels = torch.argmax(labels, dim=-1).cpu().numpy()  # バッチサイズの長さの正解ラベル配列
            correct += (pred == labels).sum().item()
    return loss_sum / float(total), correct / float(total)

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
    tokenizer = AutoTokenizer.from_pretrained(config.language_model)
    gpu_id = sys.argv[2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+ str(gpu_id) if use_cuda else "cpu")
    print(device)

    for i, seed in enumerate(config.random_seeds):
        fix_seed(seed)

        train_df, valid_df = make_dataset(seed)

        train_set     = DOLDataset(train_df, tokenizer)
        train_loader  = torch.utils.data.DataLoader(train_set, batch_size=config.batchsize, shuffle=True, num_workers=2)
        valid_set     = DOLDataset(valid_df, tokenizer)
        valid_loader  = torch.utils.data.DataLoader(valid_set , batch_size=len(valid_set)//10, shuffle=False, num_workers=2)

        out_dir = sys.argv[3] + '/seed_{}'.format(i+1)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=False)
        with open(out_dir+config.train_path, mode='w') as f:
            f.write('iterator, train_loss\n')
        with open(out_dir+config.valid_path, mode='w') as f:
            f.write('iterator, validation_loss, validation_accuracy\n')

        #setting model
        model = Classifier()
        model = model.to(device)

        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*config.n_epoch*config.warmup_rate), num_training_steps=len(train_loader)*config.n_epoch)

        adv_modules = hook_sift_layer(model, hidden_size=768)
        adv = AdversarialLearner(model, adv_modules)
        print(model)

        # train
        for e in range(config.n_epoch):
            train_loss, valid_loss, valid_acc = updater(train_loader, valid_loader, model, optimizer, scheduler, criterion, adv, device)
            print('******************************************************************')
            print("train_loss: {}, validation_loss: {}, validation_accuracy: {}".format(train_loss[-1], valid_loss[-1], valid_acc[-1]))
            print('******************************************************************')
            with open(out_dir+config.train_path, mode='a') as f:
                for i in range(len(train_loss)):
                    f.write('{}, {}\n'.format(e * len(train_loader) + i + 1, train_loss[i]))
            with open(out_dir+config.valid_path, mode='a') as f:
                for i in range(len(valid_loss)-1):
                    f.write('{}, {}, {}\n'.format(e * len(train_loader) + (i + 1), valid_loss[i], valid_acc[i]))
                f.write('E){}, {}, {}\n'.format((e + 1) * len(train_loader), valid_loss[-1], valid_acc[-1]))

            if e == 0:
                print("save best model")
                pred = valid_acc[-1]
                torch.save(model.state_dict(), out_dir+config.model_path.replace('epoch', 'best'))
            elif e == config.n_epoch - 1:
                print("save last model")
                torch.save(model.state_dict(), out_dir+config.model_path.replace('epoch', 'last'))
            else:
                if pred < valid_acc[-1]:
                    print("save best model")
                    pred = valid_acc[-1]
                    torch.save(model.state_dict(), out_dir+config.model_path.replace('epoch', 'best'))

# %% 実行部
if __name__ == '__main__':
    install()
    console = Console()
    try:
        main()
    except:
       console.print_exception()
