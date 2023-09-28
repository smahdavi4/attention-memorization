import os
import torch
import numpy as np


def analyze_attention(trainer, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    trainer.model.eval()
    auxes = []
    for x, y in test_loader:
        loss, acc, aux = trainer._evaluate_batch(x, y)
        auxes.append(aux)
    all_attn_weights = torch.cat([aux['attn_weights'] for aux in auxes], dim=0)
    first_token_attn_weights = all_attn_weights[:, :, 0]
    b, h, n = first_token_attn_weights.shape
    embeds = first_token_attn_weights.reshape(b, n * h)

    threshold = 0.99
    saturated_heads = torch.sum((first_token_attn_weights > threshold), dim=-1)

    # Save all to numpy
    np_file_path = os.path.join(save_dir, 'attention_analysis.npz')
    np.savez(
        np_file_path,
        first_token_attn_weights=first_token_attn_weights.detach().cpu().numpy(),
        embeds=embeds.detach().cpu().numpy(),
        saturated_heads=saturated_heads.detach().cpu().numpy(),
        max_attn_weight=first_token_attn_weights.max(-1).values.detach().cpu().numpy(),
        cfg={'threshold': threshold}
    )
    return np_file_path
