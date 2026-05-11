from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TSMixerConfig:
    """
    TSMixer model config for panel factor regression.

    TSMixer uses alternating MLP blocks that mix information along the
    time dimension and the feature dimension, with residual connections.

    seq_len:
        Length of historical window.
    n_features:
        Number of factor columns.
    num_blocks:
        Number of stacked TSMixer blocks.
    d_ff_time:
        Hidden dimension of the time-mixing MLP. Defaults to seq_len * 2.
    d_ff_feature:
        Hidden dimension of the feature-mixing MLP. Defaults to n_features * 2.
    dropout:
        Dropout rate.

    @JulongQuant
    """

    seq_len: int = 60
    n_features: int = 12
    num_blocks: int = 2
    d_ff_time: int | None = None
    d_ff_feature: int | None = None
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.num_blocks < 1:
            raise ValueError(
                f"num_blocks must be >= 1, got {self.num_blocks}"
            )
        if self.d_ff_time is not None and self.d_ff_time < 1:
            raise ValueError(
                f"d_ff_time must be None or >= 1, got {self.d_ff_time}"
            )
        if self.d_ff_feature is not None and self.d_ff_feature < 1:
            raise ValueError(
                f"d_ff_feature must be None or >= 1, got {self.d_ff_feature}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {self.dropout}"
            )


class _TSMixerBlock(nn.Module):
    """
    Single TSMixer block.

    Architecture:
        1. LayerNorm + Time-mixing MLP (along the time dimension, with reshape)
        2. Residual connection
        3. LayerNorm + Feature-mixing MLP (along the feature dimension)
        4. Residual connection
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_ff_time: int | None = None,
        d_ff_feature: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        d_ff_time = seq_len * 2 if d_ff_time is None else d_ff_time
        d_ff_feature = n_features * 2 if d_ff_feature is None else d_ff_feature

        # Time-mixing: applied after transposing to [B, C, L].
        self.norm_time = nn.LayerNorm(seq_len)
        self.mlp_time = nn.Sequential(
            nn.Linear(seq_len, d_ff_time),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff_time, seq_len),
            nn.Dropout(dropout),
        )

        # Feature-mixing: applied on [B, L, C].
        self.norm_feature = nn.LayerNorm(n_features)
        self.mlp_feature = nn.Sequential(
            nn.Linear(n_features, d_ff_feature),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff_feature, n_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] -> [B, L, C]
        """
        # Time-mixing with residual.
        # [B, L, C] -> [B, C, L]
        x_t = x.transpose(1, 2)
        x_t = self.norm_time(x_t)
        x_t = self.mlp_time(x_t)
        # [B, C, L] -> [B, L, C]
        x = x + x_t.transpose(1, 2)

        # Feature-mixing with residual.
        x_f = self.norm_feature(x)
        x_f = self.mlp_feature(x_f)
        x = x + x_f

        return x


class TSMixerRegressor(nn.Module):
    """
    TSMixer-style model for stock panel factor regression.

    Stacks multiple TSMixer blocks and aggregates the output to a single
    return prediction.

    Input:
        x: [batch_size, seq_len, n_features]

    Output:
        pred: [batch_size, 1]
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        num_blocks: int = 2,
        d_ff_time: int | None = None,
        d_ff_feature: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        # Instance normalization along time per feature.
        self.instance_norm = nn.LayerNorm(seq_len)

        # Stacked TSMixer blocks.
        self.blocks = nn.ModuleList([
            _TSMixerBlock(
                seq_len=seq_len,
                n_features=n_features,
                d_ff_time=d_ff_time,
                d_ff_feature=d_ff_feature,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Output head: aggregate time and feature dims.
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

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

        # [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)

        # Pass through TSMixer blocks.
        for block in self.blocks:
            x = block(x)  # [B, L, C]

        # Aggregate over time -> [B, C].
        x = x.mean(dim=1)

        # Output head -> [B, 1].
        pred = self.head(x)

        return pred


def build_tsmixer_model(config: TSMixerConfig) -> TSMixerRegressor:
    """
    Helper function for building the TSMixer model from config.
    """
    return TSMixerRegressor(
        seq_len=config.seq_len,
        n_features=config.n_features,
        num_blocks=config.num_blocks,
        d_ff_time=config.d_ff_time,
        d_ff_feature=config.d_ff_feature,
        dropout=config.dropout,
    )


if __name__ == "__main__":
    # Quick smoke test.
    config = TSMixerConfig(
        seq_len=60,
        n_features=12,
        num_blocks=2,
        dropout=0.1,
    )

    model = build_tsmixer_model(config)

    x = torch.randn(32, 60, 12)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
