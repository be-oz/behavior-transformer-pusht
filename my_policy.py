# my_policy.py
"""
Policy wrapper that uses a trained Transformer model to infer actions
from 2D agent positions in the PushT environment. The policy predicts
coarse bin indices (via classification) and residuals (via regression),
then reconstructs the final action using a learned codebook.

Author: Berke O. / https://www.linkedin.com/in/mars666/ - 2025
"""

import torch
import torch.nn as nn

# --- Static Configuration ---
D_INPUT = 2                # 2D (x, y) agent position
D_MODEL = 64               # Transformer hidden dim
NHEAD = 2                  # Multi-head attention count
NUM_LAYERS = 1             # Single transformer encoder layer
NUM_BINS = 32              # Number of coarse action bins
MODEL_PATH = "my_policy.pt"
CODEBOOK_PATH = "expert_data/action_codebook.pt"

# ----------------------------
# Transformer Architecture
# ----------------------------
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(D_INPUT, D_MODEL)                         # Input projection
        encoder = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.cls_head = nn.Linear(D_MODEL, NUM_BINS)                     # Predicts action bin index
        self.reg_head = nn.Linear(D_MODEL, D_INPUT)                      # Predicts residual refinement

    def forward(self, x):
        x = self.embed(x.unsqueeze(1))                                   # (B, 1, D_MODEL)
        x = self.transformer(x)                                          # (B, 1, D_MODEL)
        x = self.norm(x.squeeze(1))                                      # (B, D_MODEL)
        return self.cls_head(x), self.reg_head(x)                        # (B, NUM_BINS), (B, 2)

# ----------------------------
# Policy Wrapper Class
# ----------------------------
class MyPolicy:
    def __init__(self, device=None):
        # Select compute device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load trained model weights
        self.model = TransformerModel().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        # Load learned action codebook
        self.codebook = torch.load(CODEBOOK_PATH, map_location=self.device)
        assert self.codebook.shape == (NUM_BINS, D_INPUT), "Codebook shape mismatch"

    def act(self, input_dict):
        """
        Args:
            input_dict: dict with key 'observation.state' of shape (1, 2)
        Returns:
            action: torch.Tensor of shape (1, 2)
        """
        state = input_dict["observation.state"]  # Expects shape (1, 2)
        x = state.to(self.device).float()

        with torch.no_grad():
            logits, residual = self.model(x)
            idx = torch.argmax(logits, dim=-1)                         # Select highest scoring bin
            action = self.codebook[idx] + residual                     # Add residual to coarse bin

        return action
