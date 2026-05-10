from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class iTransformerConfig:
    """
    iTransformer model config for panel factor regression.

    The iTransformer applies Transformer along the variable (feature) dimension,
    treating each feature's full time series as a token, rather than
    applying attention along the time dimension.

    seq_len:
        Length of historical window.
    n_features:
        Number of factor columns.
    d_model:
        Hidden dimension for the Transformer.
    nhead:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers.
    dim_feedforward:
        Hidden dimension of the feedforward network in each encoder layer.
    dropout:
        Dropout rate.

    @JulongQuant
    """

    seq_len: int = 60
    n_features: int = 12
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


class iTransformerRegressor(nn.Module):
    """
    iTransformer-style model for stock panel factor regression.

    Each feature's time series is embedded into a token, and the Transformer
    encoder captures cross-feature interactions via self-attention.

    Input:
        x: [batch_size, seq_len, n_features]

    Output:
        pred: [batch_size, 1]
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        # Instance normalization along time per feature.
        self.instance_norm = nn.LayerNorm(seq_len)

        # Embed each feature's time series into a d_model token.
        self.embedding = nn.Linear(seq_len, d_model)

        # Learnable positional encoding for the variable dimension.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_features, d_model) * 0.02
        )

        # Transformer encoder operating on the variable dimension.
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

        # Output head.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

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

        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)

        # Instance norm along time per feature.
        x = self.instance_norm(x)  # [B, C, L]

        # Embed each feature's time series -> [B, C, d_model]
        x = self.embedding(x)

        # Add positional encoding and apply Transformer across variables.
        x = x + self.pos_embedding
        x = self.transformer(x)  # [B, C, d_model]

        # Aggregate across variables -> [B, d_model]
        x = x.mean(dim=1)

        # Output head -> [B, 1]
        pred = self.head(x)

        return pred


def build_itransformer_model(
    config: iTransformerConfig,
) -> iTransformerRegressor:
    """
    Helper function for building the iTransformer model from config.
    """
    return iTransformerRegressor(
        seq_len=config.seq_len,
        n_features=config.n_features,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )


if __name__ == "__main__":
    # Quick smoke test.
    config = iTransformerConfig(
        seq_len=60,
        n_features=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    model = build_itransformer_model(config)

    x = torch.randn(32, 60, 12)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
