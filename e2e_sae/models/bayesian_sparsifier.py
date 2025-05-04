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
    def __init__(self):
        """
        Initialize HardConcrete module. Parameters are now passed to forward.
        """
        super().__init__()
        # Beta, stretch_limits (l, r), epsilon are now passed to forward

    def forward(
        self,
        logits: torch.Tensor,
        beta: float,
        l: float,
        r: float,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Sample from the Hard Concrete distribution using the reparameterization trick.

        Args:
            logits: Logits parameter (alpha) for the distribution. Shape: (*, num_features)
            beta: Temperature parameter. Controls the sharpness of the distribution.
            l: Lower bound of the stretch interval.
            r: Upper bound of the stretch interval.
            epsilon: Small constant for numerical stability.

        Returns:
            z: Sampled values (hard-thresholded in [0, 1]). Shape: (*, num_features)
        """
        if not (l < 0.0 and r > 1.0):
            # Warning: L0 penalty calculation (done elsewhere) might be incorrect if l >= 0 or r <= 1
            pass

        if self.training:
            # Sample u ~ Uniform(0, 1)
            u = torch.rand_like(logits)
            # Transform to Concrete variable s ~ Concrete(logits, beta) = Sigmoid((log(u) - log(1-u) + logits) / beta)
            s = torch.sigmoid((torch.log(u + epsilon) - torch.log(1.0 - u + epsilon) + logits) / beta)
            # Stretch s to (l, r)
            s_stretched = s * (r - l) + l
            # Apply hard threshold (clamp to [0, 1]) -> z ~ HardConcrete(logits, beta)
            z = torch.clamp(s_stretched, min=0.0, max=1.0)
        else:
            # Evaluation mode: use deterministic output
            # Use the clamped stretched sigmoid mean approximation
            s = torch.sigmoid(logits)
            s_stretched = s * (r - l) + l
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
        initial_beta: float, # Initial temperature for Hard Concrete
        stretch_limits: tuple[float, float] = (-0.1, 1.1), # Stretch limits for Hard Concrete
        init_decoder_orthogonal: bool = True,
    ):
        """Initialize the SAE with Hard Concrete gates.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components (and Hard Concrete gates)
            initial_beta: Initial temperature for the Hard Concrete distribution. This will be annealed during training.
            stretch_limits: Stretch limits (l, r) for Hard Concrete. Must have l < 0 and r > 1.
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
        """
        super().__init__()
        # Encoder outputs logits for gating and pre-activation for magnitude
        self.encoder = nn.Linear(input_size, 2 * n_dict_components, bias=True)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=True)

        self.n_dict_components = n_dict_components
        self.input_size = input_size

        # Register beta as a buffer to allow updates during training without being a model parameter
        self.register_buffer("beta", torch.tensor(initial_beta))
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"

        # Instantiate the Hard Concrete sampler dynamically in forward pass
        self.sampler = HardConcrete() # Instantiated without parameters now

        if init_decoder_orthogonal:
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """
        Pass input through the encoder to get logits and pre-magnitude values.
        Sample gates z using Hard Concrete from logits with the current beta.
        Calculate magnitudes using ReLU on pre-magnitude.
        Combine gates and magnitudes for final coefficients c.
        Reconstruct using the normalized decoder.

        Returns:
            x_hat: Reconstructed input.
            c: Final coefficients (gate * magnitude).
            logits: The logits produced by the encoder for the gates (used for L0 penalty calculation).
            beta: The beta value used for sampling.
            l: The lower stretch limit used.
            r: The upper stretch limit used.
        """
        # Get encoder output (logits and pre-magnitude combined)
        encoder_out = self.encoder(x)

        # Split encoder output into logits and pre-magnitude
        logits, pre_magnitude = torch.chunk(encoder_out, 2, dim=-1)

        # Sample gates z from Hard Concrete distribution using logits and current beta
        current_beta = self.beta.item() # Get current beta value from buffer
        z = self.sampler(logits, beta=current_beta, l=self.l, r=self.r) # Shape: (batch_size, n_dict_components)

        # Calculate magnitude using ReLU on pre-magnitude
        # magnitude_c = F.relu(pre_magnitude) # Shape: (batch_size, n_dict_components)

        # Combine gate and magnitude for final coefficients
        c = z * pre_magnitude # Shape: (batch_size, n_dict_components)

        # Reconstruct using the dictionary elements and final coefficients
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)

        # Return logits and HC params for L0 loss calculation
        return x_hat, c, logits, current_beta, self.l, self.r


    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        # Normalize columns (dim=0) of the weight matrix
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
