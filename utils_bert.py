import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
import joblib
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
print(f"torch.__version__: {torch.__version__}")

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--model_name', type=str, default="hfl/chinese-roberta-wwm-ext",
                    help='预训练模型名字')
parser.add_argument('--max_len', type=int, default=256, help='文本最大长度')
parser.add_argument('--trn_fold', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='分折个数')
parser.add_argument('--log_prefix', type=str, default="train", help='日志文件名称')
args = parser.parse_args()

print(args.trn_fold)
print(args.log_prefix)
print(args.max_len)
lb=joblib.load('models/models2/lb_category.joblib')

# ====================================================
# CFG:参数配置
# ====================================================
class Config:
    # 配置
    apex = False
    seed = 42  # 随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0  # 进程个数
    print_freq = 100  # 打印频率
    debug = False
    train = True
    predict = True

    # 预训练模型
    model_type = 'bert'
    model_name = args.model_name


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 数据相关
    task_name = ''  # 比赛任务名称
    dataset_name = ''  # 数据集名称
    text_col = 'text'  # 文本列名
    target_col = 'category'  # label列名
    target_size = 5  # 类别个数
    max_len = 154  # 最大长度
    batch_size = 32  # 模型运行批处理大小
    n_fold = 10  # cv折数
    trn_folds = [0]  # 需要用的折数

    # 模型训练超参数
    epochs = 7  # 训练轮次
    lr = 2e-5  # 训练过程中的最大学习率
    eps = 1e-6
    betas = (0.9, 0.999)
    # warmup_proportion = 0.1  # 学习率预热比例
    num_warmup_steps = 0.1
    T_0 = 2  # CosineAnnealingWarmRestarts
    min_lr = 1e-6
    warmup_ratio = 0.1
    weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    gradient_accumulation_steps = 1  # 梯度累加
    max_grad_norm = 1.0  # 梯度剪切

    # 目录设置
    # 目录设置
    log_prefix = args.log_prefix  # 模型输出目录
    output_dir = './models/models2'  # 模型输出目录
    save_prefix = model_name.split("/")[-1]  # 保存文件前缀
    log_prefix = log_prefix + '_' + save_prefix

    # trick
    use_fgm = True  # fgm pgd
    use_pgd = False  # fgm pgd

    pgd_k = 3
    use_ema = True
    use_rdrop = True
    use_multidrop = False

    use_noisy = False
    use_mixout = False

    # 损失函数
    criterion = 'CrossEntropyLoss'  # - ['LabelSmoothing', 'FocalLoss','FocalLossPro','FocalCosineLoss', 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss']
    smoothing = 0.01
    scheduler = 'linear'  # ['linear', 'cosine']

    # score_type
    score_type = 'f1_macro'


CFG = Config()

# os.makedirs(CFG.output_dir, exist_ok=True)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.seed)






def clean_txt(txt):
    try:
        txt = txt.replace(' ', '')
        txt = txt.replace('　', '')
        txt = txt.replace('\r', '')
        txt = txt.replace('\n', '')
        txt = txt.replace('“', '"')
        txt = txt.replace('”', '"')
        txt = txt.replace(',', '，')

        if txt[-1] == '.':
            txt[-1] == '。'
    except Exception as e:
        txt = txt.encode('utf-8', 'replace').decode('utf-8')
    txt = ''.join(txt.split())
    return txt





class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        #         print(encoding['input_ids'])
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size, is_shuffle=False):
    ds = CustomDataset(
        texts=df[CFG.text_col].values,
        labels=df[CFG.target_col].values,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_shuffle,
        pin_memory=True,
        # num_workers=4  # windows多线程
    )


class Custom_bert(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        self.linear = nn.Linear(self.config.hidden_size, n_classes)
        self._init_weights(self.linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        logits = self.linear(mean_embeddings)
        return logits


def get_predictions(model, data_loader):
    model = model.eval()

    prediction_probs = []

    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader), desc="inference"):
            input_ids = d["input_ids"].to(CFG.device)
            attention_mask = d["attention_mask"].to(CFG.device)
            token_type_ids = d["token_type_ids"].to(CFG.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            probs = outputs.softmax(1)
            prediction_probs.extend(probs)
    prediction_probs = torch.stack(prediction_probs).cpu().numpy()
    return prediction_probs


def inference(test):
    # test = pd.read_csv('demo_zhongmei2022.csv')
    test['text'] = test['text'].apply(clean_txt)
    print(test.shape)
    test['category'] = 0
    test_data_loader = create_data_loader(test, CFG.tokenizer, CFG.max_len, CFG.batch_size, False)
    probs = np.zeros((len(test), CFG.target_size))
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_folds:
            model = Custom_bert(model_name=CFG.model_name, n_classes=CFG.target_size)
            path = f'{CFG.output_dir}/{CFG.save_prefix}_best_model_fold{fold}.bin'
            state = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            model.eval()
            model.to(CFG.device)

            print(next(model.parameters()).device)

            prediction_probs = get_predictions(model, test_data_loader)
            probs += prediction_probs / len(CFG.trn_folds)
    np.save(f'{CFG.output_dir}/{CFG.save_prefix}_probs.npy', probs)
    print(probs)
    labels = np.argmax(probs, axis=1)

    test['category'] = lb.inverse_transform(labels)
    print(test['category'].value_counts())
    # test.to_csv(f'models2/result_demo_zhongmei2022.csv', index=False)
    return test

# if __name__ == '__main__':
#     inference()

"""
content_full
5    174141
4     48290
1     11698
2      8973
3      7639
"""

"""
raw_text
5    174141
4     48290
1     11698
2      8973
3      7639
"""

"""
text
5    171750
4     53775
2     10178
1      9424
3      5614
"""
