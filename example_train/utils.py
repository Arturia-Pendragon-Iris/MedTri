import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
# from blosc_RW import *
from PIL import Image
from explanation_augment import annotate_text_with_levenshtein
from counterfact_augment import remix_two_anatomies


# 你自己的 CT 读取函数
# from your_lib import load_blosc


class MIMICTextCLIPDataset(Dataset):
    def __init__(self,
                 device,
                 excel_path="/mnt/data_2/X-Ray/MIMIC/MIMIC.xlsx",
                 ct_dir="/mnt/data_2/X-Ray/MIMIC/webdataset",
                 ratio=1):
        super().__init__()
        self.info = np.array(pd.read_excel(excel_path))
        # patient_id -> 行索引 的映射，加速查找
        self.pid_to_index = {
            str(row[0]): idx for idx, row in enumerate(self.info)
        }

        self.file_list = sorted(glob.glob(f"{ct_dir}/*"))
        self.file_list = self.file_list[:int(ratio * len(self.file_list))]

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
            use_fast=True,
        )

        self.device = device

        print(f"[CTTextCLIPDataset] Total CT samples: {len(self.file_list)}")
        print(f"[CTTextCLIPDataset] Total text rows: {self.info.shape[0]}")

    def __len__(self):
        return len(self.file_list)

    def _process_ct(self, filepath):
        file = np.random.choice(os.listdir(filepath))
        arr = np.array(Image.open(os.path.join(filepath, file)).convert("L"), dtype=np.float32) / 255.0

        return torch.tensor(arr[np.newaxis], dtype=torch.float32)

    def _encode_text(self, text: str):
        """返回 input_ids: [seq_len]."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        # [1, seq_len] -> [seq_len]
        return enc["input_ids"].squeeze(0)

    def __getitem__(self, index):
        # ---------------- 图像 ----------------
        filepath = self.file_list[index]
        patient_id = filepath.split("/")[-1]  # 根据你文件名的实际格式修改

        ct_tensor = self._process_ct(filepath)

        # ---------------- 正样本文本 ----------------
        if patient_id not in self.pid_to_index:
            raise KeyError(f"Patient id {patient_id} not found in Excel info")

        row_idx = self.pid_to_index[patient_id]
        (text_1, text_2, text_3) = (str(self.info[row_idx, 1]),
                                    str(self.info[row_idx, 2]),
                                    str(self.info[row_idx, 3]))
        (input_ids_1, input_ids_2, input_ids_3) = (self._encode_text(text_1),
                                                   self._encode_text(text_2),
                                                   self._encode_text(text_3))
        # print(patient_id, text_1, text_2, input_ids_1, input_ids_2)
        # exit()
        return {
            "image": ct_tensor,
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "input_ids_3": input_ids_3,
        }


class MIMIC_ExDataset(Dataset):
    def __init__(self,
                 device,
                 excel_path="/mnt/data_2/X-Ray/MIMIC/MIMIC.xlsx",
                 ct_dir="/mnt/data_2/X-Ray/MIMIC/webdataset",
                 ratio=1):
        super().__init__()
        self.info = np.array(pd.read_excel(excel_path))
        # patient_id -> 行索引 的映射，加速查找
        self.pid_to_index = {
            str(row[0]): idx for idx, row in enumerate(self.info)
        }

        self.file_list = sorted(glob.glob(f"{ct_dir}/*"))
        self.file_list = self.file_list[:int(ratio * len(self.file_list))]

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
            use_fast=True,
        )

        self.device = device

        with open("/data_backup/Project/Text_transfer/diseases_description.json", "r", encoding="utf-8") as f:
            self.disease_ex = json.load(f)

        print(f"[CTTextCLIPDataset] Total CT samples: {len(self.file_list)}")
        print(f"[CTTextCLIPDataset] Total text rows: {self.info.shape[0]}")

    def __len__(self):
        return len(self.file_list)

    def _process_ct(self, filepath):
        file = np.random.choice(os.listdir(filepath))
        arr = np.array(Image.open(os.path.join(filepath, file)).convert("L"), dtype=np.float32) / 255.0

        return torch.tensor(arr[np.newaxis], dtype=torch.float32)

    def _encode_text(self, text: str):
        """返回 input_ids: [seq_len]."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        # [1, seq_len] -> [seq_len]
        return enc["input_ids"].squeeze(0)

    def __getitem__(self, index):
        # ---------------- 图像 ----------------
        filepath = self.file_list[index]
        patient_id = filepath.split("/")[-1]  # 根据你文件名的实际格式修改

        ct_tensor = self._process_ct(filepath)

        # ---------------- 正样本文本 ----------------
        if patient_id not in self.pid_to_index:
            raise KeyError(f"Patient id {patient_id} not found in Excel info")

        row_idx = self.pid_to_index[patient_id]
        text = str(self.info[row_idx, 2])
        annotated_text = annotate_text_with_levenshtein(text, self.disease_ex)

        input_ids = self._encode_text(annotated_text)

        # print(patient_id, text_1, text_2, input_ids_1, input_ids_2)
        # exit()
        return {
            "image": ct_tensor,
            "input_ids": input_ids,
        }


class MIMIC_CounterDataset(Dataset):
    def __init__(self,
                 device,
                 excel_path="/mnt/data_2/X-Ray/MIMIC/MIMIC.xlsx",
                 ct_dir="/mnt/data_2/X-Ray/MIMIC/webdataset",
                 ratio=1):
        super().__init__()
        self.info = np.array(pd.read_excel(excel_path))
        # patient_id -> 行索引 的映射，加速查找
        self.pid_to_index = {
            str(row[0]): idx for idx, row in enumerate(self.info)
        }

        self.file_list = sorted(glob.glob(f"{ct_dir}/*"))
        self.file_list = self.file_list[:int(ratio * len(self.file_list))]

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
            use_fast=True,
        )

        self.device = device

        with open("/data_backup/Project/Text_transfer/anatomy.json", "r", encoding="utf-8") as f:
            self.options = json.load(f)

        print(f"[CTTextCLIPDataset] Total CT samples: {len(self.file_list)}")
        print(f"[CTTextCLIPDataset] Total text rows: {self.info.shape[0]}")

    def __len__(self):
        return len(self.file_list)

    def _process_ct(self, filepath):
        file = np.random.choice(os.listdir(filepath))
        arr = np.array(Image.open(os.path.join(filepath, file)).convert("L"), dtype=np.float32) / 255.0

        return torch.tensor(arr[np.newaxis], dtype=torch.float32)

    def _encode_text(self, text: str):
        """返回 input_ids: [seq_len]."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        # [1, seq_len] -> [seq_len]
        return enc["input_ids"].squeeze(0)

    def __getitem__(self, index):
        # ---------------- 图像 ----------------
        filepath = self.file_list[index]
        patient_id = filepath.split("/")[-1]  # 根据你文件名的实际格式修改

        ct_tensor = self._process_ct(filepath)

        # ---------------- 正样本文本 ----------------
        if patient_id not in self.pid_to_index:
            raise KeyError(f"Patient id {patient_id} not found in Excel info")

        row_idx = self.pid_to_index[patient_id]
        text = str(self.info[row_idx, 2])
        counter_text, _ = remix_two_anatomies(text, self.options)
        # print(counter_text)

        input_ids = self._encode_text(text)
        counter_input_ids = self._encode_text(counter_text)

        # print(patient_id, text_1, text_2, input_ids_1, input_ids_2)
        # exit()
        return {
            "image": ct_tensor,
            "input_ids": input_ids,
            "counter_input_ids": counter_input_ids
        }
