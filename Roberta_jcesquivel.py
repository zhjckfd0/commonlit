# Imports

import os
import random
import numpy as np
import pandas as pd
import glob
import re
import gc; gc.enable()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, SequentialSampler, DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, logging

import transformers

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error

# Constants

SEED = 42

HIDDEN_SIZE = 1024
MAX_LEN = 300

INPUT_DIR = 'D:\pycharm_projects\CommonLit'
BASELINE_DIR = '../input/commonlit-readability-models'
MODEL_DIR = "D:\pycharm_projects\CommonLit\models\\roberta-transformers-pytorch\\roberta-large"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

# Utility functions

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# Data

submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
test.head()

# Dataset

class CLRPDataset(Dataset):
     def __init__(self, texts, tokenizer):
         self.texts = texts
         self.tokenizer = tokenizer

     def __len__(self):
         return len(self.texts)

     def __getitem__(self, idx):
         encode = self.tokenizer(
             self.texts[idx],
             padding='max_length',
             max_length=MAX_LEN,
             truncation=True,
             add_special_tokens=True,
             return_attention_mask=True,
             return_tensors='pt'
         )
         return encode

# Model

class MeanPoolingModel(nn.Module):

     def __init__(self, model_name):
         super().__init__()

         config = AutoConfig.from_pretrained(model_name)
         self.model = AutoModel.from_pretrained(model_name, config=config)
         self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
         self.linear = nn.Linear(HIDDEN_SIZE, 1)
         self.loss = nn.MSELoss()

     def forward(self, input_ids, attention_mask, labels=None):

         outputs = self.model(input_ids, attention_mask)
         last_hidden_state = outputs[0]
         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
         sum_mask = input_mask_expanded.sum(1)
         sum_mask = torch.clamp(sum_mask, min=1e-9)
         mean_embeddings = sum_embeddings / sum_mask
         norm_mean_embeddings = self.layer_norm(mean_embeddings)
         logits = self.linear(norm_mean_embeddings)

         preds = logits.squeeze(-1).squeeze(-1)

         if labels is not None:
             loss = self.loss(preds.view(-1).float(), labels.view(-1).float())
             return loss
         else:
             return preds

def predict(df, model):
     ds = CLRPDataset(df.excerpt.tolist(), TOKENIZER)
     dl = DataLoader(
         ds,
         batch_size=BATCH_SIZE,
         shuffle=False,
         pin_memory=False
     )

     model.to(DEVICE)
     model.eval()
     model.zero_grad()

     predictions = []
     for batch in tqdm(dl):
         inputs = {key: val.reshape(val.shape[0], -1).to(DEVICE) for key, val in batch.items()}
         outputs = model(**inputs)
         predictions.extend(outputs.detach().cpu().numpy().ravel())

     return predictions

 # Calculate predictions of each fold and average them

fold_predictions = []
for path in glob.glob(BASELINE_DIR + '/*.ckpt'):
    model = MeanPoolingModel(MODEL_DIR)
    model.load_state_dict(torch.load(path))
    fold = int(re.match(r'.*_f_?(\d)_.*', path).group(1))
    print(f'*** fold {fold} ***')
    y_pred = predict(test, model)
    fold_predictions.append(y_pred)

    # Free memory
    del model
    gc.collect()

predictions = np.mean(fold_predictions, axis=0)

submission['target'] = predictions
submission.to_csv('submission111.csv', index=False)
submission.head()