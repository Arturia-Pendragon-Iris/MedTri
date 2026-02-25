import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer
from transformers import AutoModel
from example_train.modules import *


class MIMIC_SwinTransformer(nn.Module):
    def __init__(self,
                 in_channels=2,
                 feature_size=24,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 projection_dim=512,
                 temperature_init=0.07):
        super().__init__()

        # ----------------------------
        # 1️⃣ Image Encoder (MONAI SwinTransformer)
        # ----------------------------
        self.image_encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(5, 5),
            patch_size=(4, 4),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            spatial_dims=2,
            use_v2=True,
        )

        img_embed_dim = feature_size * 16
        text_embed_dim = 768

        self.image_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # ----------------------------
        # 3️⃣ Temperature 参数（与 CLIP 相同）
        # ----------------------------
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    # ============================================================
    # encode_image / encode_text：标准 CLIP 风格接口
    # ============================================================
    def encode_image(self, ct: torch.Tensor) -> torch.Tensor:
        """
        ct: [B, C, D, H]
        return: L2-normalized image embeddings [B, proj_dim]
        """
        B = ct.size(0)
        feats = self.image_encoder(ct)

        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats

        x = x.view(B, x.shape[1], -1).mean(dim=-1)  # [B, C]

        x = self.image_proj(x)  # [B, proj_dim]
        x = F.normalize(x, dim=-1)  # L2 normalize
        return x

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids:     [B, L]
        attention_mask:[B, L] or None
        return: L2-normalized text embeddings [B, proj_dim]
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token embedding
        txt = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        txt = self.text_proj(txt)  # [B, proj_dim]
        txt = F.normalize(txt, dim=-1)  # L2 normalize
        return txt

    # ============================================================
    # forward：标准 CLIP InfoNCE（in-batch negatives）
    # ============================================================
    def forward(self,
                ct: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None
                ):
        img_emb = self.encode_image(ct)  # [B, D]
        txt_emb = self.encode_text(input_ids, attention_mask)  # [B, D]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # [B, B]：image->text
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        # [B, B]：text->image
        logits_per_text = logit_scale * txt_emb @ img_emb.t()

        batch_size = img_emb.size(0)
        targets = torch.arange(batch_size, device=img_emb.device)

        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2.0

        return {
            "loss_value": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
        }


class MIMIC_ViT(nn.Module):
    def __init__(self,
                 in_channels=1,
                 mid_chans=256,
                 projection_dim=512,
                 temperature_init=0.07):
        super().__init__()

        # ----------------------------
        # 1️⃣ Image Encoder (MONAI SwinTransformer)
        # ----------------------------
        self.image_encoder = ImageEncoderViT(img_size=512,
                                             in_chans=in_channels,
                                             out_chans=mid_chans,
                                             patch_size=16,
                                             use_rel_pos=False,
                                             rel_pos_zero_init=True,
                                             global_attn_indexes=(2, 5, 8, 11))

        img_embed_dim = mid_chans
        text_embed_dim = 768

        self.image_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    def encode_image(self, ct: torch.Tensor) -> torch.Tensor:
        """
        ct: [B, C, D, H]
        return: L2-normalized image embeddings [B, proj_dim]
        """
        B = ct.size(0)
        feats = self.image_encoder(ct)

        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats

        # print(x.shape)

        # x: [B, C, D', H'] -> [B, C, D, 'H']
        x = x.view(B, x.shape[1], -1).mean(dim=-1)  # [B, C]

        x = self.image_proj(x)  # [B, proj_dim]
        x = F.normalize(x, dim=-1)  # L2 normalize
        return x

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids:     [B, L]
        attention_mask:[B, L] or None
        return: L2-normalized text embeddings [B, proj_dim]
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token embedding
        txt = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        txt = self.text_proj(txt)  # [B, proj_dim]
        txt = F.normalize(txt, dim=-1)  # L2 normalize
        return txt

    def forward(self,
                ct: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None
                ):
        img_emb = self.encode_image(ct)  # [B, D]
        txt_emb = self.encode_text(input_ids, attention_mask)  # [B, D]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        logits_per_text = logit_scale * txt_emb @ img_emb.t()

        batch_size = img_emb.size(0)
        targets = torch.arange(batch_size, device=img_emb.device)

        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2.0

        return {
            "loss_value": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
        }


class MIMIC_Conv(nn.Module):
    def __init__(self,
                 in_channels=1,
                 mid_chans=256,
                 projection_dim=512,
                 temperature_init=0.07):
        super().__init__()

        # ----------------------------
        # 1️⃣ Image Encoder (MONAI SwinTransformer)
        # ----------------------------
        self.image_encoder = ConvNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 128, mid_chans], in_chans=in_channels)

        img_embed_dim = mid_chans
        text_embed_dim = 768

        self.image_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    def encode_image(self, ct: torch.Tensor) -> torch.Tensor:
        """
        ct: [B, C, D, H]
        return: L2-normalized image embeddings [B, proj_dim]
        """
        B = ct.size(0)
        feats = self.image_encoder(ct)  # 可能是 list/tuple 或单个 tensor

        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats

        # x: [B, C, D', H'] -> [B, C, D, 'H']
        x = x.view(B, x.shape[1], -1).mean(dim=-1)  # [B, C]

        x = self.image_proj(x)  # [B, proj_dim]
        x = F.normalize(x, dim=-1)  # L2 normalize
        return x

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids:     [B, L]
        attention_mask:[B, L] or None
        return: L2-normalized text embeddings [B, proj_dim]
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token embedding
        txt = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        txt = self.text_proj(txt)  # [B, proj_dim]
        txt = F.normalize(txt, dim=-1)  # L2 normalize
        return txt

    # ============================================================
    # forward：标准 CLIP InfoNCE（in-batch negatives）
    # ============================================================
    def forward(self,
                ct: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None
                ):
        img_emb = self.encode_image(ct)  # [B, D]
        txt_emb = self.encode_text(input_ids, attention_mask)  # [B, D]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # [B, B]：image->text
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        # [B, B]：text->image
        logits_per_text = logit_scale * txt_emb @ img_emb.t()

        batch_size = img_emb.size(0)
        targets = torch.arange(batch_size, device=img_emb.device)

        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2.0

        return {
            "loss_value": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
        }


class MIMIC_SwinCounter(nn.Module):
    def __init__(self,
                 in_channels=2,
                 feature_size=24,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 projection_dim=512,
                 temperature_init=0.07):
        super().__init__()

        self.image_encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(5, 5),
            patch_size=(4, 4),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            spatial_dims=2,
            use_v2=True,
        )

        img_embed_dim = feature_size * 16
        text_embed_dim = 768

        self.image_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Temperature（与 CLIP 相同）
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    def encode_image(self, ct: torch.Tensor) -> torch.Tensor:
        """
        ct: [B, C, D, H]
        return: L2-normalized image embeddings [B, proj_dim]
        """
        B = ct.size(0)
        feats = self.image_encoder(ct)

        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats

        x = x.view(B, x.shape[1], -1).mean(dim=-1)  # GAP -> [B, C]
        x = self.image_proj(x)  # [B, proj_dim]
        x = F.normalize(x, dim=-1)
        return x

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        txt = outputs.last_hidden_state[:, 0, :]  # CLS [B, 768]
        txt = self.text_proj(txt)  # [B, proj_dim]
        txt = F.normalize(txt, dim=-1)
        return txt

    def forward(self,
                ct: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                neg_input_ids: torch.Tensor = None,
                neg_attention_mask: torch.Tensor = None):

        img_emb = self.encode_image(ct)  # [B, D]
        txt_emb = self.encode_text(input_ids, attention_mask)  # [B, D]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        if neg_input_ids is not None:
            if neg_input_ids.dim() == 2:  # [B, L]
                # 构造 mask
                if neg_attention_mask is None:
                    neg_attention_mask = (neg_input_ids != 0).long()
                neg_emb = self.encode_text(neg_input_ids, neg_attention_mask)  # [B, D]
            elif neg_input_ids.dim() == 3:  # [B, K, L]
                B, K, L = neg_input_ids.shape
                neg_input_ids_flat = neg_input_ids.view(B * K, L)
                if neg_attention_mask is None:
                    neg_attention_mask = (neg_input_ids_flat != 0).long()
                else:
                    neg_attention_mask = neg_attention_mask.view(B * K, L)
                neg_emb = self.encode_text(neg_input_ids_flat, neg_attention_mask)  # [B*K, D]
            else:
                raise ValueError("neg_input_ids must be [B, L] or [B, K, L].")
            txt_bank = torch.cat([txt_emb, neg_emb], dim=0)  # [B + B*K?, D]
        else:
            txt_bank = txt_emb

        logits_per_image = logit_scale * (img_emb @ txt_bank.t())  # [B, B + BN?]

        B = img_emb.size(0)
        targets = torch.arange(B, device=img_emb.device)
        loss_i = F.cross_entropy(logits_per_image, targets)

        logits_per_text = logit_scale * (txt_emb @ img_emb.t())  # [B, B]
        loss_t = F.cross_entropy(logits_per_text, targets)

        loss = 0.5 * (loss_i + loss_t)

        return {
            "loss_value": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
        }


class MIMIC_ViTCounter(nn.Module):
    def __init__(self,
                 in_channels=2,
                 projection_dim=512,
                 mid_chans=256,
                 temperature_init=0.07):
        super().__init__()

        self.image_encoder = ImageEncoderViT(img_size=512,
                                             in_chans=in_channels,
                                             out_chans=mid_chans,
                                             patch_size=16,
                                             use_rel_pos=False,
                                             rel_pos_zero_init=True,
                                             global_attn_indexes=(2, 5, 8, 11))

        img_embed_dim = 256
        text_embed_dim = 768

        self.image_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Temperature（与 CLIP 相同）
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    def encode_image(self, ct: torch.Tensor) -> torch.Tensor:
        """
        ct: [B, C, D, H]
        return: L2-normalized image embeddings [B, proj_dim]
        """
        B = ct.size(0)
        feats = self.image_encoder(ct)

        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats

        x = x.view(B, x.shape[1], -1).mean(dim=-1)  # GAP -> [B, C]
        x = self.image_proj(x)  # [B, proj_dim]
        x = F.normalize(x, dim=-1)
        return x

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        txt = outputs.last_hidden_state[:, 0, :]  # CLS [B, 768]
        txt = self.text_proj(txt)  # [B, proj_dim]
        txt = F.normalize(txt, dim=-1)
        return txt

    def forward(self,
                ct: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                neg_input_ids: torch.Tensor = None,
                neg_attention_mask: torch.Tensor = None):

        img_emb = self.encode_image(ct)  # [B, D]
        txt_emb = self.encode_text(input_ids, attention_mask)  # [B, D]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        if neg_input_ids is not None:
            if neg_input_ids.dim() == 2:  # [B, L]
                # 构造 mask
                if neg_attention_mask is None:
                    neg_attention_mask = (neg_input_ids != 0).long()
                neg_emb = self.encode_text(neg_input_ids, neg_attention_mask)  # [B, D]
            elif neg_input_ids.dim() == 3:  # [B, K, L]
                B, K, L = neg_input_ids.shape
                neg_input_ids_flat = neg_input_ids.view(B * K, L)
                if neg_attention_mask is None:
                    neg_attention_mask = (neg_input_ids_flat != 0).long()
                else:
                    neg_attention_mask = neg_attention_mask.view(B * K, L)
                neg_emb = self.encode_text(neg_input_ids_flat, neg_attention_mask)  # [B*K, D]
            else:
                raise ValueError("neg_input_ids must be [B, L] or [B, K, L].")
            txt_bank = torch.cat([txt_emb, neg_emb], dim=0)  # [B + B*K?, D]
        else:
            txt_bank = txt_emb

        logits_per_image = logit_scale * (img_emb @ txt_bank.t())  # [B, B + BN?]

        # 标签仍然指向前 B 个正文本
        B = img_emb.size(0)
        targets = torch.arange(B, device=img_emb.device)
        loss_i = F.cross_entropy(logits_per_image, targets)

        logits_per_text = logit_scale * (txt_emb @ img_emb.t())  # [B, B]
        loss_t = F.cross_entropy(logits_per_text, targets)

        loss = 0.5 * (loss_i + loss_t)

        return {
            "loss_value": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
        }
