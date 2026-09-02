from torch import nn
import torch.nn.functional as F
import torch

#跨模态对比：正样本对是图像前景区域和对应文本，负样本对是前景区域和同批次中其他与其对应文本内容不同的文本
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.15, grid_division=5, hard_weight_factor=2.0, semantic_threshold=0.6):
        super().__init__()
        self.temp = temperature
        self.grid_division = grid_division
        self.hard_weight_factor = hard_weight_factor  # 困难负样本权重倍数
        self.semantic_threshold = semantic_threshold  # 语义相似度阈值
        self.text_proj = nn.Linear(768, 1024)

    def forward(self, patch_embeds, text_embeds, region_mask, text_list):
        B, C, H, W = patch_embeds.shape

        # 处理文本特征
        if text_embeds.dim() == 3:
            text_embeds = text_embeds.mean(dim=2)
        text_feat = F.normalize(self.text_proj(text_embeds), dim=-1)

        # 计算文本语义相似度矩阵
        text_sim_matrix = torch.mm(text_feat, text_feat.T)

        # 调整mask到特征图尺寸
        mask = F.interpolate(region_mask.unsqueeze(1).float(), (H, W), mode='nearest')
        mask = mask.squeeze(1) > 0.5

        grid_size_h = max(H // self.grid_division, 1)
        grid_size_w = max(W // self.grid_division, 1)

        total_loss = 0
        valid_regions = 0

        # 滑动窗口遍历
        for i in range(0, H, grid_size_h):
            for j in range(0, W, grid_size_w):
                h_end = min(i + grid_size_h, H)
                w_end = min(j + grid_size_w, W)

                # 提取区域特征
                region_feat = patch_embeds[:, :, i:h_end, j:w_end]
                region_mask_patch = mask[:, i:h_end, j:w_end]

                # 有效性检查
                valid_batch = (region_mask_patch.sum(dim=(1, 2)) / ((h_end - i) * (w_end - j)) > 0.1)
                if not valid_batch.any():
                    continue

                # 特征聚合
                valid_feat = region_feat[valid_batch]
                valid_mask = region_mask_patch[valid_batch].unsqueeze(1)
                masked_feat = (valid_feat * valid_mask).sum(dim=(2, 3)) / (valid_mask.sum(dim=(2, 3)) + 1e-6)
                masked_feat = F.normalize(masked_feat, dim=-1)

                region_text = text_feat[valid_batch]
                sim_matrix = torch.mm(masked_feat, region_text.T) / self.temp

                if text_list is not None:
                    valid_texts = [text_list[idx] for idx, v in enumerate(valid_batch) if v]
                    valid_indices = torch.where(valid_batch)[0]

                    # 获取当前有效batch的文本相似度子矩阵
                    text_sim_submatrix = text_sim_matrix[valid_indices][:, valid_indices]

                    # 使用加权softmax计算损失
                    loss = self.weighted_softmax_loss(sim_matrix, text_sim_submatrix, valid_texts)
                else:
                    # 如果没有文本列表，回退到原始损失
                    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                    loss = F.cross_entropy(sim_matrix, labels)

                total_loss += loss
                valid_regions += 1

        return total_loss / (valid_regions + 1e-6) if valid_regions > 0 else 0.0

    def weighted_softmax_loss(self, sim_matrix, text_sim_matrix, valid_texts):
        """加权softmax实现语义感知对比损失"""
        B = sim_matrix.size(0)

        # 构建基础mask: 只保留正样本和内容不同的负样本
        base_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for row in range(len(valid_texts)):
            for col in range(len(valid_texts)):
                if row == col or valid_texts[row] != valid_texts[col]:
                    base_mask[row, col] = True

        # 创建权重矩阵，初始权重为1
        weight_matrix = torch.ones_like(sim_matrix)

        # 识别并加权困难负样本
        for i in range(B):
            for j in range(B):
                if i != j and base_mask[i, j]:  # 只处理负样本位置
                    semantic_sim = text_sim_matrix[i, j].item()

                    # 如果是语义相似但实际不同的困难负样本
                    if semantic_sim > self.semantic_threshold:
                        # 根据语义相似度线性调整权重
                        # 相似度越高，权重越大
                        weight_factor = 1.0 + (self.hard_weight_factor - 1.0) * (
                                (semantic_sim - self.semantic_threshold) / (1.0 - self.semantic_threshold)
                        )
                        weight_matrix[i, j] = weight_factor

        # 应用mask并计算加权softmax
        masked_sim = sim_matrix.clone()
        masked_sim[~base_mask] = float('-inf')

        # 加权softmax计算
        # 公式: softmax = exp(sim * weight) / sum(exp(sim * weight))
        # 但我们使用: (exp(sim) * weight) / sum(exp(sim) * weight)
        weighted_exp = torch.exp(masked_sim) * weight_matrix

        # 计算softmax分母
        softmax_denom = weighted_exp.sum(dim=1, keepdim=True)

        # 计算softmax概率
        softmax_probs = weighted_exp / (softmax_denom + 1e-8)

        # 计算负对数似然损失
        pos_probs = softmax_probs.diag()
        loss = -torch.log(pos_probs + 1e-8).mean()

        return loss
