from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

ckpt_dir = "./checkpoint-7086"  # 改成你的checkpoint目录或最终output目录
instruction_template = (
   "Convert the following radiology findings (text may include final diagnosis) into a structured summary. Format: each line anatomy: findings; diagnosis-category."
    "Only include anatomies present in the text; no speculation; keep each line to one sentence; total ≤250 words."
    "Use objective imaging terms; ban words: suggestive of, could represent, likely, recommend correlation."
    "Input:{src} Output:"
)

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

src_text = """
The trachea was in the midline of both main bronchi and no obstructive pathology was detected in the lumen. The mediastinum could not be evaluated optimally in the non-contrast examination. As far as can be seen; mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Accessory spleen with a diameter of 7.5 mm was observed in the anterior neighborhood of the upper pole of the spleen. Vertebral corpus heights were preserved in bone structures in the study area. Fracture lines were observed in the anterior part of the right 1st, 4th, and 5th ribs, and at the level of the 2nd and 3rd rib costochondral junction. There is also a contusion in the anterior part of the 6th rib.	There was no finding in favor of infection in the lung parenchyma. Fracture-contusion lines in the ribs defined in the right hemithorax

"""

inp = instruction_template.format(src=src_text)

enc = tokenizer(inp, return_tensors="pt", truncation=True).to(device)
gen_ids = model.generate(
    **enc,
    max_new_tokens=512,         # 与训练时 label_max_len 对齐或略小
    num_beams=4,
    no_repeat_ngram_size=3,
)
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
