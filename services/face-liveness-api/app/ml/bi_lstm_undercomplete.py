"""BiLSTM Undercomplete Autoencoder.

Provenance
----------
The only temporal checkpoint shipped with the project is
``saved_models/ABC_BiLSTM_Undercomplete_AE.pt``. Its state dict contains
bidirectional encoder/decoder gates (``*_reverse`` keys), a
``latent (256, 1024)`` projection, an ``expand (1024, 256)`` projection and an
autoregressive decoder whose input width is ``1664 == 1408 + 256``.

No class in the original ``models/lstm_models.py`` matches that shape layout
(``LSTMAutoencoder`` is unidirectional; ``BiLSTMVariationalAutoencoder`` uses
``fc_mu``/``fc_logvar`` and a non-autoregressive decoder). This module
reconstructs the missing architecture directly from the checkpoint layout.

Verification (performed during implementation, see docs/IMPLEMENTATION_REPORT.md):
the reconstructed model loads the checkpoint with ``strict=True`` -
0 missing keys, 0 unexpected keys - and reproduces the historical decisions
on the bundled Oulu samples (real -> 0.035 REAL, printer spoof -> 0.996 SPOOF).

One flag cannot be recovered from the checkpoint: ``reverse_target``. Both
values produce numerically near-identical anomaly scores (the autoregressive
flip largely cancels). We default to ``True`` to match the sibling
``LSTMAutoencoder`` convention used by the original training code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMUndercompleteAE(nn.Module):
    """Bidirectional LSTM undercomplete autoencoder with autoregressive decode.

    Encoder: BiLSTM(input=1408, hidden=512) -> concat last fwd/bwd hidden (1024)
    Latent:  Linear(1024 -> 256)  (undercomplete: 256 < 1408)
    Decoder: autoregressive BiLSTM(input=1408+256, hidden=512), init hidden from
             Linear(256 -> 1024); each step feeds [prev_frame, latent].
    Output:  Linear(1024 -> 1408)
    """

    def __init__(
        self,
        input_size: int = 1408,
        hidden_size: int = 512,
        latent_size: int = 256,
        num_layers: int = 1,
        reverse_target: bool = True,
    ):
        super().__init__()
        if latent_size >= input_size:
            raise ValueError("Must be undercomplete AE (latent_size < input_size)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.reverse_target = reverse_target

        # Encoder - bidirectional, same dims as checkpoint
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.latent = nn.Linear(hidden_size * 2, latent_size)

        # Decoder - bidirectional + autoregressive, same dims as checkpoint
        self.expand = nn.Linear(latent_size, hidden_size * 2)
        self.decoder = nn.LSTM(
            input_size=input_size + latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = nn.Linear(hidden_size * 2, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        return self.latent(torch.cat([forward_hidden, backward_hidden], dim=1))

    def _decode_autoregressive(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = latent.size(0)
        device = latent.device

        # Initial hidden state from latent, split across the two directions.
        h0 = self.expand(latent).view(batch_size, 2, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = torch.zeros_like(h0)
        hidden_state = (h0, c0)

        prev_frame = torch.zeros(batch_size, 1, self.input_size, device=device)
        outputs = []

        for _ in range(seq_len):
            step_input = torch.cat([prev_frame, latent.unsqueeze(1)], dim=-1)
            decoder_output, hidden_state = self.decoder(step_input, hidden_state)
            frame_output = self.output_layer(decoder_output)
            outputs.append(frame_output)
            # Eval-only path: no teacher forcing; feed prediction back.
            prev_frame = frame_output.detach()

        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstruction = self._decode_autoregressive(latent, seq_len)
        if self.reverse_target:
            reconstruction = torch.flip(reconstruction, dims=[1])
        return reconstruction
