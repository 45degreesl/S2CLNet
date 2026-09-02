from torch import nn
import torch.nn.functional as F
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.15, grid_division=5, hard_weight_factor=2.0, semantic_threshold=0.6):
        super().__init__()
        self.temp = temperature
        self.grid_division = grid_division
        self.hard_weight_factor = hard_weight_factor
        self.semantic_threshold = semantic_threshold
        self.text_proj = nn.Linear(768, 1024)

    def forward(self, patch_embeds, text_embeds, region_mask, text_list):
        B, C, H, W = patch_embeds.shape

        if text_embeds.dim() == 3:
            text_embeds = text_embeds.mean(dim=2)
        text_feat = F.normalize(self.text_proj(text_embeds), dim=-1)

        text_sim_matrix = torch.mm(text_feat, text_feat.T)

        mask = F.interpolate(region_mask.unsqueeze(1).float(), (H, W), mode='nearest')
        mask = mask.squeeze(1) > 0.5

        grid_size_h = max(H // self.grid_division, 1)
        grid_size_w = max(W // self.grid_division, 1)

        total_loss = 0
        valid_regions = 0

        for i in range(0, H, grid_size_h):
            for j in range(0, W, grid_size_w):
                h_end = min(i + grid_size_h, H)
                w_end = min(j + grid_size_w, W)

                region_feat = patch_embeds[:, :, i:h_end, j:w_end]
                region_mask_patch = mask[:, i:h_end, j:w_end]

                valid_batch = (region_mask_patch.sum(dim=(1, 2)) / ((h_end - i) * (w_end - j)) > 0.1)
                if not valid_batch.any():
                    continue

                valid_feat = region_feat[valid_batch]
                valid_mask = region_mask_patch[valid_batch].unsqueeze(1)
                masked_feat = (valid_feat * valid_mask).sum(dim=(2, 3)) / (valid_mask.sum(dim=(2, 3)) + 1e-6)
                masked_feat = F.normalize(masked_feat, dim=-1)

                region_text = text_feat[valid_batch]
                sim_matrix = torch.mm(masked_feat, region_text.T) / self.temp

                if text_list is not None:
                    valid_texts = [text_list[idx] for idx, v in enumerate(valid_batch) if v]
                    valid_indices = torch.where(valid_batch)[0]

                    text_sim_submatrix = text_sim_matrix[valid_indices][:, valid_indices]

                    loss = self.weighted_softmax_loss(sim_matrix, text_sim_submatrix, valid_texts)
                else:
                    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                    loss = F.cross_entropy(sim_matrix, labels)

                total_loss += loss
                valid_regions += 1

        return total_loss / (valid_regions + 1e-6) if valid_regions > 0 else 0.0

    def weighted_softmax_loss(self, sim_matrix, text_sim_matrix, valid_texts):
        B = sim_matrix.size(0)

        base_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for row in range(len(valid_texts)):
            for col in range(len(valid_texts)):
                if row == col or valid_texts[row] != valid_texts[col]:
                    base_mask[row, col] = True

        weight_matrix = torch.ones_like(sim_matrix)

        for i in range(B):
            for j in range(B):
                if i != j and base_mask[i, j]:
                    semantic_sim = text_sim_matrix[i, j].item()

                    if semantic_sim > self.semantic_threshold:
                        weight_factor = 1.0 + (self.hard_weight_factor - 1.0) * (
                                (semantic_sim - self.semantic_threshold) / (1.0 - self.semantic_threshold)
                        )
                        weight_matrix[i, j] = weight_factor

        masked_sim = sim_matrix.clone()
        masked_sim[~base_mask] = float('-inf')

        weighted_exp = torch.exp(masked_sim) * weight_matrix

        softmax_denom = weighted_exp.sum(dim=1, keepdim=True)

        softmax_probs = weighted_exp / (softmax_denom + 1e-8)

        pos_probs = softmax_probs.diag()
        loss = -torch.log(pos_probs + 1e-8).mean()

        return loss
