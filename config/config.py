##### Training Data
# OLID
train_data_EN_path = '/home/usrs/tomoki.fujihara.p3/OLID/English/preprocess/cleaned/olid-training-v1.0.tsv'
train_data_transJP_path  = '/home/usrs/tomoki.fujihara.p3/OLID/Japanese/XMAT/with_preprocess/DeepL/trainset_translate.tsv'
# SOLID
train_data_AR_path = '/home/usrs/tomoki.fujihara.p3/SOLID/Arabic/preprocess/cleaned/offenseval-ar-training-v1.tsv'
train_data_DA_path = '/home/usrs/tomoki.fujihara.p3/SOLID/Danish/preprocess/cleaned/offenseval-da-training-v1.tsv'
train_data_TR_path = '/home/usrs/tomoki.fujihara.p3/SOLID/Turkish/preprocess/cleaned/offenseval-tr-training-v1.tsv'


##### Test Data
# test_data_path  = '/home/usrs/tomoki.fujihara.p3/OLID/English/preprocess/cleaned/testset-levela.tsv'
test_data_path   = '/home/usrs/tomoki.fujihara.p3/OLID/Japanese/XMAT/with_preprocess/DeepL/testset-levela.tsv'

test_unlabeled_offjp_path  = '/home/usrs/tomoki.fujihara.p3/JPOLID/unlabeled_dataset/offensive/tool_name/new_dataset.tsv'
test_unlabeled_notoffjp_path  = '/home/usrs/tomoki.fujihara.p3/JPOLID/unlabeled_dataset/not_offensive/tool_name/dataset.tsv'

# labeled_by_fujihara = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_by_fujihara/test_data_en.csv'
labeled_by_fujihara = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_by_fujihara/test_data.csv'

japanese_offensive_language_dataset = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_dataset/japanese_offensive_language_dataset.tsv'

only_label_A_dataset = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_dataset/only_label_a.tsv'
only_label_B_dataset = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_dataset/only_label_b.tsv'

split_4_groups_dataset = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_dataset/split_4_groups.tsv'
absolute_label_dataset = '/home/usrs/tomoki.fujihara.p3/JPOLID/labeled_dataset/absolute_label.tsv'

# BERTの論文でFine-tuningの際に推奨されている値
n_epoch = 3
max_token_len = 256 # Github参照: https://github.com/google-research/bert#out-of-memory-issues
batchsize = 16
warmup_rate = 0.1 # イテレーション全体の元の10%
lr = 2e-5
dropout_rate = 0.1

output_class_num = 2
reinit_n_layers = 1

language_model = 'studio-ousia/luke-japanese-base-lite'
random_seeds = [2, 10, 42, 541, 3407]

train_path = '/train_loss.csv'
valid_path = '/valid_loss.csv'
test_path = '/test_last_model.csv'
model_path = '/model_epoch.pth'
test_model_path = '/model_last.pth'
test_result_path = '/test_result'
