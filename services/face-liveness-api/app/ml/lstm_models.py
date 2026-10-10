import torch
import torch.nn as nn
import random


class LSTMAutoencoder(nn.Module):
    """LSTM Undercomplete Autoencoder - EXACTLY same as training"""
    
    def __init__(
        self,
        input_size=1408,
        hidden_size=512,
        latent_size=256,
        num_layers=1,
        reverse_target=True,
        autoregressive_decode=True,
        teacher_forcing_ratio=0.5
    ):
        super().__init__()

        assert latent_size < input_size, "Must be undercomplete AE"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.reverse_target = reverse_target
        self.autoregressive_decode = autoregressive_decode
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder - same as training
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.latent = nn.Linear(hidden_size, latent_size)

        # Decoder - same as training
        self.expand = nn.Linear(latent_size, hidden_size)

        if autoregressive_decode:
            self.decoder = nn.LSTM(
                input_size=input_size + latent_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_last = hidden[-1]
        latent = self.latent(hidden_last)
        return latent

    def _decode_autoregressive(self, latent, seq_len, target=None):
        batch_size = latent.size(0)
        device = latent.device

        h0 = self.expand(latent).unsqueeze(0)
        h0 = h0.repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        hidden_state = (h0, c0)

        prev_frame = torch.zeros(batch_size, 1, self.input_size, device=device)
        outputs = []

        for t in range(seq_len):
            step_input = torch.cat([prev_frame, latent.unsqueeze(1)], dim=-1)
            decoder_output, hidden_state = self.decoder(step_input, hidden_state)
            frame_output = self.output_layer(decoder_output)
            outputs.append(frame_output)

            use_teacher_forcing = (
                self.training and target is not None and
                random.random() < self.teacher_forcing_ratio
            )

            if use_teacher_forcing:
                prev_frame = target[:, t:t+1, :]
            else:
                prev_frame = frame_output.detach()

        return torch.cat(outputs, dim=1)

    def forward(self, x):
        seq_len = x.size(1)
        latent = self.encode(x)

        if self.reverse_target:
            target = torch.flip(x, dims=[1])
        else:
            target = x

        if self.autoregressive_decode:
            reconstruction = self._decode_autoregressive(latent, seq_len, target)
        else:
            reconstruction = self._decode_repeat_vector(latent, seq_len)

        if self.reverse_target:
            reconstruction = torch.flip(reconstruction, dims=[1])

        return reconstruction


class BiLSTMVariationalAutoencoder(nn.Module):
    """BiLSTM Variational Autoencoder - EXACTLY same as training"""
    
    def __init__(
        self,
        input_size=1408,
        hidden_size=512,
        latent_size=256,
        num_layers=1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        encoder_output_size = hidden_size * 2

        # Encoder - BiLSTM - same as training
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # μ (mean) - same as training
        self.fc_mu = nn.Linear(encoder_output_size, latent_size)

        # log(σ²) - same as training
        self.fc_logvar = nn.Linear(encoder_output_size, latent_size)

        # Decoder expansion - same as training
        self.expand = nn.Linear(latent_size, hidden_size * 2)

        # Decoder - BiLSTM - same as training
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.output_layer = nn.Linear(hidden_size * 2, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder - same as training
        _, (hidden, _) = self.encoder(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Latent distribution - same as training
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        latent = self.reparameterize(mu, logvar)

        # Decoder - same as training
        decoder_input = self.expand(latent).unsqueeze(1)
        decoder_input = decoder_input.repeat(1, x.size(1), 1)

        decoder_output, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(decoder_output)

        return reconstruction


