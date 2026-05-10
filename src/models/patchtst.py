from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PatchTSTConfig:
    """
    PatchTST model config for panel factor regression.

    PatchTST divides each feature's time series into patches (subseries),
    embeds each patch, and processes them with a Transformer encoder in a
    channel-independent fashion (weights shared across features).

    seq_len:
        Length of historical window.
    n_features:
        Number of factor columns.
    patch_len:
        Length of each patch.
    stride:
        Stride between consecutive patches.
    d_model:
        Hidden dimension for the Transformer.
    nhead:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers.
    dim_feedforward:
        Hidden dimension of the feedforward network.
    dropout:
        Dropout rate.

    @JulongQuant
    """

    seq_len: int = 60
    n_features: int = 12
    patch_len: int = 8
    stride: int = 4
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.patch_len < 1:
            raise ValueError(f"patch_len must be >= 1, got {self.patch_len}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.patch_len > self.seq_len:
            raise ValueError(
                f"patch_len ({self.patch_len}) must not exceed seq_len ({self.seq_len})"
            )
        remaining = (self.seq_len - self.patch_len) % self.stride
        if remaining != 0:
            raise ValueError(
                f"(seq_len - patch_len) % stride must be 0 to avoid "
                f"discarding trailing timesteps, but got remainder={remaining} "
                f"(seq_len={self.seq_len}, patch_len={self.patch_len}, "
                f"stride={self.stride})"
            )
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if self.nhead < 1:
            raise ValueError(f"nhead must be >= 1, got {self.nhead}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.dim_feedforward < 1:
            raise ValueError(
                f"dim_feedforward must be >= 1, got {self.dim_feedforward}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {self.dropout}"
            )


class PatchTSTRegressor(nn.Module):
    """
    PatchTST-style model for stock panel factor regression.

    Each feature's time series is independently divided into patches,
    embedded, and processed by a shared Transformer encoder. The outputs
    are aggregated across patches and features to produce a single return
    prediction.

    Input:
        x: [batch_size, seq_len, n_features]

    Output:
        pred: [batch_size, 1]
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        if (seq_len - patch_len) % stride != 0:
            raise ValueError(
                f"(seq_len - patch_len) % stride must be 0, but got "
                f"remainder={(seq_len - patch_len) % stride} "
                f"(seq_len={seq_len}, patch_len={patch_len}, stride={stride})"
            )

        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches < 1:
            raise ValueError(
                f"seq_len={seq_len}, patch_len={patch_len}, stride={stride} "
                f"result in {self.num_patches} patches (need >= 1)"
            )

        # Shared patch embedding and positional encoding.
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Shared Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Instance normalization for each feature's time series.
        self.instance_norm = nn.LayerNorm(seq_len)

        # Output head: aggregate across patches and features.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)

        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor with shape [B, L, C]

        Returns
        -------
        Tensor with shape [B, 1]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, L, C], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {x.shape[1]}"
            )
        if x.shape[2] != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, got {x.shape[2]}"
            )

        B, L, C = x.shape

        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)

        # Instance norm along time per feature.
        x = self.instance_norm(x)  # [B, C, L]

        # Unfold each feature's time series into patches.
        # [B, C, L] -> [B, C, num_patches, patch_len]
        patches = x.unfold(
            dimension=2, size=self.patch_len, step=self.stride
        )
        _, _, num_patches, patch_len = patches.shape

        # Flatten batch and channel dims: shared weights across features.
        # [B, C, num_patches, patch_len] -> [B*C, num_patches, patch_len]
        patches = patches.reshape(B * C, num_patches, patch_len)

        # Embed patches -> [B*C, num_patches, d_model]
        patches = self.patch_embedding(patches)
        patches = patches + self.pos_embedding

        # Transformer encoder -> [B*C, num_patches, d_model]
        patches = self.transformer(patches)

        # Aggregate across patches -> [B*C, d_model]
        patches = patches.mean(dim=1)

        # Reshape back: [B, C, d_model] -> aggregate across features -> [B, d_model]
        patches = patches.reshape(B, C, -1).mean(dim=1)

        # Output head -> [B, 1]
        pred = self.head(patches)

        return pred


def build_patchtst_model(config: PatchTSTConfig) -> PatchTSTRegressor:
    """
    Helper function for building the PatchTST model from config.
    """
    return PatchTSTRegressor(
        seq_len=config.seq_len,
        n_features=config.n_features,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )


if __name__ == "__main__":
    # Quick smoke test.
    config = PatchTSTConfig(
        seq_len=60,
        n_features=12,
        patch_len=8,
        stride=4,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    model = build_patchtst_model(config)

    x = torch.randn(32, 60, 12)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
