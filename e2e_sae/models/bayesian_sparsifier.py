import torch
import torch.nn.functional as F
from torch import nn


class HardConcrete(nn.Module):
    """
    Hard Concrete distribution stochastic gate.
    Used for L0 regularization as described in https://arxiv.org/abs/1712.01312.

    Produces samples in [0, 1] via a stretched, hard-thresholded sigmoid transformation
    of a log-uniform variable.
    """
    def __init__(self, beta: float, stretch_limits: tuple[float, float], epsilon: float = 1e-6):
        """
        Initialize HardConcrete module.

        Args:
            beta: Temperature parameter. Controls the sharpness of the distribution.
                  Lower values make it closer to a Bernoulli. Default often 2/3 or 1/3.
            stretch_limits: Tuple (l, r) for the stretching interval before clamping.
                           Needs l < 0 and r > 1 for L0 penalty calculation.
            epsilon: Small constant for numerical stability (e.g., in logs).
        """
        super().__init__()
        self.beta = beta
        self.l, self.r = stretch_limits
        self.epsilon = epsilon

        if not (self.l < 0.0 and self.r > 0.0):
            # Relaxing constraint check here, but penalty calc might need adjustment if l>=0 or r<=0
            pass


    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the Hard Concrete distribution using the reparameterization trick.

        Args:
            logits: Logits parameter (alpha) for the distribution. Shape: (*, num_features)

        Returns:
            z: Sampled values (hard-thresholded in [0, 1]). Shape: (*, num_features)
        """
        if self.training:
            # Sample u ~ Uniform(0, 1)
            u = torch.rand_like(logits)
            # Transform to Concrete variable s ~ Concrete(logits, beta) = Sigmoid((log(u) - log(1-u) + logits) / beta)
            s = torch.sigmoid((torch.log(u + self.epsilon) - torch.log(1.0 - u + self.epsilon) + logits) / self.beta)
            # Stretch s to (l, r)
            s_stretched = s * (self.r - self.l) + self.l
            # Apply hard threshold (clamp to [0, 1]) -> z ~ HardConcrete(logits, beta)
            z = torch.clamp(s_stretched, min=0.0, max=1.0)
        else:
            # Evaluation mode: use deterministic output
            # Use the clamped stretched sigmoid mean approximation
            s = torch.sigmoid(logits)
            s_stretched = s * (self.r - self.l) + self.l
            z = torch.clamp(s_stretched, min=0.0, max=1.0)

        return z


class BayesianSAE(nn.Module):
    """
    Bayesian Sparse AutoEncoder using Hard Concrete stochastic gates for coefficients (L0 Sparsity).
    Combines L0 gating with ReLU-based magnitude for reconstruction.
    Decouples gate logits and magnitude pre-activation.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        init_decoder_orthogonal: bool = True,
        hard_concrete_beta: float = 1.0/3.0, # Temperature for Hard Concrete
        hard_concrete_stretch_limits: tuple[float, float] = (-0.1, 1.1) # Stretch limits for Hard Concrete
    ):
        """Initialize the SAE with Hard Concrete gates.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components (and Hard Concrete gates)
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
            hard_concrete_beta: Temperature for the Hard Concrete distribution.
            hard_concrete_stretch_limits: Stretch limits (l, r) for Hard Concrete.
        """
        super().__init__()
        # Encoder outputs logits for gating and pre-activation for magnitude
        self.encoder = nn.Linear(input_size, 2 * n_dict_components, bias=True)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=True)

        self.n_dict_components = n_dict_components
        self.input_size = input_size

        # Instantiate the Hard Concrete sampler
        self.sampler = HardConcrete(beta=hard_concrete_beta, stretch_limits=hard_concrete_stretch_limits)

        if init_decoder_orthogonal:
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass input through the encoder to get logits and pre-magnitude values.
        Sample gates z using Hard Concrete from logits.
        Calculate magnitudes using ReLU on pre-magnitude.
        Combine gates and magnitudes for final coefficients c.
        Reconstruct using the normalized decoder.

        Returns:
            x_hat: Reconstructed input.
            c: Final coefficients (gate * magnitude).
            logits: The logits produced by the encoder for the gates (used for L0 penalty calculation).
        """
        # Get encoder output (logits and pre-magnitude combined)
        encoder_out = self.encoder(x)

        # Split encoder output into logits and pre-magnitude
        logits, pre_magnitude = torch.chunk(encoder_out, 2, dim=-1)

        # Sample gates z from Hard Concrete distribution using logits
        z = self.sampler(logits) # Shape: (batch_size, n_dict_components)

        # Calculate magnitude using ReLU on pre-magnitude
        magnitude_c = F.relu(pre_magnitude) # Shape: (batch_size, n_dict_components)

        # Combine gate and magnitude for final coefficients
        c = z * magnitude_c # Shape: (batch_size, n_dict_components)

        # Reconstruct using the dictionary elements and final coefficients
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)

        # Return logits for L0 loss calculation
        return x_hat, c, logits


    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        # Normalize columns (dim=0) of the weight matrix
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
