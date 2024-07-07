import torch
import torch.nn as nn
import os
from transformers import CLIPVisionModel
from torchvision import transforms
from PIL import Image
from ..hunyuan_models import attention, apply_rotary_emb
from .resampler import Resampler

class CrossAttention_IPA(nn.Module):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        cross_attn,
        ip_kv_proj,
        ip_k_norm
    ):
        super().__init__()
        self.cross_attn = cross_attn
        self.ip_kv_proj = ip_kv_proj
        self.ip_k_norm = ip_k_norm
        self.ip_y = None
        self.org_cross_attn_ratio = 1.0
        self.ip_cross_attn_ratio = 1.0

    def set_attn_mode(self, mode):
        self.cross_attn.attn_mode = mode

    def set_ip_emb(self, tensor):
        assert tensor.dim() == 3
        self.ip_y = tensor

    def clear_ip_emb(self):
        self.ip_y = None

    def __getattr__(self, name):
        try:
            return super(CrossAttention_IPA, self).__getattr__(name)
        except AttributeError:
            return getattr(self.cross_attn, name)
    
    def forward(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num_heads * head_dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // num_heads), RoPE for image
        """
        ip_y = self.ip_y
        assert ip_y is not None
        b, s1, _ = x.shape  # [b, s1, D]
        _, s2, _ = y.shape  # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        kv = self.kv_proj(y).view(
            b, s2, 2, self.num_heads, self.head_dim
        )  # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2)  # [b, s2, h, d]
        q = self.q_norm(q).to(q)  # [b, s1, h, d]
        k = self.k_norm(k).to(k)  # [b, s2, h, d]
        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            assert qq.shape == q.shape, f"qq: {qq.shape}, q: {q.shape}"
            q = qq  # [b, s1, h, d]
        # kv = torch.stack([k, v], dim=2)  # [b, s1, 2, h, d]
        # context = self.inner_attn(q, kv)  # [b, s1, h, d]
        context = attention(q, k, v, self.head_dim, self.attn_drop, mode=self.attn_mode)

        _, s_ip, _ = ip_y.shape
        ip_kv = self.ip_kv_proj(ip_y).view(
            b, s_ip, 2, self.num_heads, self.head_dim
        ) 
        ip_k, ip_v = ip_kv.unbind(dim=2)  # [b, s2, h, d]
        ip_k = self.ip_k_norm(ip_k).to(ip_k)
        ip_context = attention(q, ip_k, ip_v, self.head_dim, self.attn_drop, mode=self.attn_mode)
        context = context * self.org_cross_attn_ratio + ip_context * self.ip_cross_attn_ratio
        context = context.reshape(b, s1, -1)  # [b, s1, D]

        out = self.out_proj(context)
        out = self.proj_drop(out)

        out_tuple = (out,)
        return out_tuple


class IPAModel:
    def __init__(self, unet_blocks, device, dtype, clip_path=None, ip_cross_attn_path=None, resampler_path=None):
        self.device = device
        self.dtype = dtype
        if clip_path is None:
            self.image_encoder = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        else:
            self.image_encoder = CLIPVisionModel.from_pretrained(clip_path)
        self.image_encoder.to(self.device, self.dtype)
        self.embedding_dim = self.image_encoder.config.hidden_size
        self.OUTPUT_DIM = 1024
        self.NUM_QUERIES = 8
        self.NUM_LATENTS_MEAN_POOLED = 4  # 0 for no mean pooling (previous behavior)
        self.APPLY_POS_EMB = True  # False for no positional embeddings (previous behavior)
        self.image_proj_model = Resampler(
            dim=1024,
            depth=2,
            dim_head=64,
            heads=16,
            num_queries=self.NUM_QUERIES,
            embedding_dim=self.embedding_dim,
            output_dim=self.OUTPUT_DIM,
            ff_mult=2,
            max_seq_len=257,
            apply_pos_emb=self.APPLY_POS_EMB,
            num_latents_mean_pooled=self.NUM_LATENTS_MEAN_POOLED,
        )
        if resampler_path is not None:
            self.image_proj_model.load_state_dict(torch.load(resampler_path, map_location="cpu"))
        self.image_proj_model.to(self.device, self.dtype)
        self.unet_blocks = unet_blocks
        cross_attn_num = len(self.unet_blocks)
        ip_cross_attn_module_list = []
        for i in range(cross_attn_num):
            org_cross_attn = self.unet_blocks[i].attn2
            factory_kwargs = {"device": self.device, "dtype": self.dtype}
            ip_kv_proj = nn.Linear(1024, 2 * org_cross_attn.qdim, bias=org_cross_attn.qkv_bias, **factory_kwargs)
            ip_kv_proj.to(self.device, self.dtype)
            ip_k_norm = (
                nn.LayerNorm(org_cross_attn.head_dim, elementwise_affine=True, eps=1e-6)
                if org_cross_attn.qk_norm
                else nn.Identity()
            )
            ip_k_norm.to(self.device, self.dtype)
            self.unet_blocks[i].attn2 = CrossAttention_IPA(org_cross_attn, ip_kv_proj, ip_k_norm)
            ip_cross_attn_module_list.append([ip_kv_proj, ip_k_norm])
        self.ip_cross_attn_module_list = ip_cross_attn_module_list
        if ip_cross_attn_path is not None:
            self.ip_cross_attn_load_state_dict(ip_cross_attn_path)

        def image_preprocess(pil_image):
            transform = transforms.Compose([
                transforms.Resize([224, 224]),  # 先将图片缩放至256
                transforms.ToTensor(),  # 转换为Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ])
            return transform(pil_image.convert('RGB'))
        self.negative = False
        self.image_preprocessor = image_preprocess

    def ip_cross_attn_load_state_dict(self, path):
        state_dict = torch.load(path, map_location=self.device)
        for i in range(len(state_dict)):
            self.ip_cross_attn_module_list[i][0].load_state_dict(state_dict[i]["ip_kv_proj"])
            self.ip_cross_attn_module_list[i][1].load_state_dict(state_dict[i]["ip_k_norm"])

    def ip_cross_attn_save_state_dict(self, path, acc_wrapped=False):
        state_dict = {}
        if acc_wrapped:
            for i in range(len(self.ip_cross_attn_module_list)):
                state_dict[i] = {"ip_kv_proj":self.ip_cross_attn_module_list[i][0].modules.state_dict(), "ip_k_norm":self.ip_cross_attn_module_list[i][1].module.state_dict()}
            torch.save(state_dict, path)
        else:
            for i in range(len(self.ip_cross_attn_module_list)):
                state_dict[i] = {"ip_kv_proj":self.ip_cross_attn_module_list[i][0].state_dict(), "ip_k_norm":self.ip_cross_attn_module_list[i][1].state_dict()}
            torch.save(state_dict, path)
        
    
    def preprocess_image_list(self, image_list):
        image_tensor_list = [self.image_preprocessor(image) for image in image_list]
        return torch.stack(image_tensor_list).to(self.device, self.dtype)

    @staticmethod
    def load_image_path_list(image_path_list):
        return [Image.open(image_path) for image_path in image_path_list]

    def set_ip_scale(self, scale1, scale2):
        for block in self.unet_blocks:
            block.attn2.org_cross_attn_ratio = scale1
            block.attn2.ip_cross_attn_ratio = scale2

    def clear_ip_emb(self):
        for block in self.unet_blocks:
            block.attn2.clear_ip_emb()
    
    def set_ip_emb(self, tensor):
        for block in self.unet_blocks:
            block.attn2.set_ip_emb(tensor)
    
    def set_ip_emb_with_image_tensor(self, tensor):
        self.set_ip_emb(self.get_image_prompt_emb(tensor))

    def get_negative_clip_emb(self, bs):
        negative_emb = torch.zeros([1, 257, self.embedding_dim], device=self.device, dtype=self.dtype)
        return negative_emb.repeat(bs, 1, 1)

    def get_image_prompt_emb(self, image_tensor):
        if isinstance(image_tensor, list):
            if image_tensor[0].dim() == 3:
                image_tensor = torch.stack(image_tensor)
            elif image_tensor[0].dim() == 4:
                image_tensor = torch.cat(image_tensor, dim=0)
        with torch.no_grad():
            clip_vision_emb = self.image_encoder(image_tensor, output_hidden_states=True).hidden_states[-2]
        if self.negative:
            clip_vision_emb = torch.cat([self.get_negative_clip_emb(clip_vision_emb.shape[0]), clip_vision_emb], dim=0)
        image_prompt_emb = self.image_proj_model(clip_vision_emb)
        return image_prompt_emb
    
    def set_ip_emb_with_path_list(self, image_path_list):
        tensor = self.preprocess_image_list(self.load_image_path_list(image_path_list))
        self.set_ip_emb(self.get_image_prompt_emb(tensor))

    def save_state_dict_to_folder(self, folder_path, acc_wrapped=False):
        ip_cross_attn_path = os.path.join(folder_path, "ip_cross_attn.pt")
        resampler_path = os.path.join(folder_path, "resampler.pt")
        if acc_wrapped:
            torch.save(self.image_proj_model.modules.state_dict(), resampler_path)
        else:
            torch.save(self.image_proj_model.state_dict(), resampler_path)
        self.ip_cross_attn_save_state_dict(ip_cross_attn_path, acc_wrapped=acc_wrapped)

    def load_state_dict_from_folder(self, folder_path):
        ip_cross_attn_path = os.path.join(folder_path, "ip_cross_attn.pt")
        resampler_path = os.path.join(folder_path, "resampler.pt")
        self.image_proj_model.load_state_dict(torch.load(resampler_path, map_location=self.device))
        self.ip_cross_attn_load_state_dict(ip_cross_attn_path)

    def get_trainable_param(self):
        trainable_param_list = list(self.image_proj_model.parameters())
        for ip_cross_attn_module in self.ip_cross_attn_module_list:
            trainable_param_list.extend(list(ip_cross_attn_module[0].parameters()))
            trainable_param_list.extend(list(ip_cross_attn_module[1].parameters()))
        return trainable_param_list
        
    
    def set_requires_grad(self, requires_grad=True):
        self.image_proj_model.requires_grad_(requires_grad)
        for ip_cross_attn_module in self.ip_cross_attn_module_list:
            ip_cross_attn_module[0].requires_grad_(requires_grad)
            ip_cross_attn_module[1].requires_grad_(requires_grad)
        self.image_encoder.requires_grad_(False)

    def all_module_need_to_train_list(self):
        module_list = [self.image_proj_model]
        for modules in self.ip_cross_attn_module_list:
            module_list.extend(modules)
        return module_list
    
    def to(self, device=None, dtype=None):
        if device is not None:
            self.image_encoder.to(device)
            self.image_proj_model.to(device)
            for ip_cross_attn_module in self.ip_cross_attn_module_list:
                ip_cross_attn_module[0].to(device)
                ip_cross_attn_module[1].to(device)
            self.device = device
        if dtype is not None:
            self.image_encoder.to(dtype)
            self.image_proj_model.to(dtype)
            for ip_cross_attn_module in self.ip_cross_attn_module_list:
                ip_cross_attn_module[0].to(dtype)
                ip_cross_attn_module[1].to(dtype)
            self.dtype = dtype