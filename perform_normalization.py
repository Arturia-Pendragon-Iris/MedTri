from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

ckpt_dir = "./checkpoint-7086"

instruction_template = (
    "Convert the following radiology findings (text may include final diagnosis) into a structured summary. "
    "Format: each line anatomy: findings; diagnosis-category."
    "Only include anatomies present in the text; no speculation; keep each line to one sentence; total ≤250 words."
    "Use objective imaging terms; ban words: suggestive of, could represent, likely, recommend correlation."
    "Input:{src} Output:"
)

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    ckpt_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

device = "cuda"
model.to(device)

torch.set_grad_enabled(False)

df_raw = pd.read_excel("/data_backup/Project/Text_transfer/evaluation.xlsx")

re_table = []
time_list = []

batch_size = 16
save_every_batches = 50

for batch_start in tqdm(range(0, len(df_raw), batch_size)):

    start = time()
    batch = df_raw.iloc[batch_start: batch_start + batch_size]

    ids = batch.iloc[:, 0].tolist()
    batch.iloc[:, 1] = batch.iloc[:, 1].astype(str)

    src_texts = (batch.iloc[:, 1]).tolist()

    inputs = [instruction_template.format(src=src) for src in src_texts]

    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    gen_ids = model.generate(
        **enc,
        max_new_tokens=256,
        num_beams=4,
        no_repeat_ngram_size=3,
    )

    outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    end = time()

    time_list.append([end - start])
#     for _id, out in zip(ids, outputs):
#         re_table.append([_id, out])
#
#     # 4) 定期保存，减少 Excel I/O
#     if (batch_start // batch_size) % save_every_batches == 0 and batch_start > 0:
#         df_tmp = pd.DataFrame(re_table, columns=["id", "summary"])
#         df_tmp.to_excel("/data/temp/RE_1.xlsx", index=False)
#
# df_final = pd.DataFrame(re_table, columns=["id", "summary"])
# df_final.to_excel("/data/temp/RE_1.xlsx", index=False)

time_list = np.array(time_list)
print(np.mean(time_list), np.std(time_list))