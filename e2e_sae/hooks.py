from typing import Any, NamedTuple

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

from e2e_sae.models.sparsifiers import SAE
from e2e_sae.models.bayesian_sparsifier import BayesianSAE


class CacheActs(NamedTuple):
    input: Float[torch.Tensor, "... dim"]


class SAEActs(NamedTuple):
    input: Float[torch.Tensor, "... dim"]
    c: Float[torch.Tensor, "... c"]
    output: Float[torch.Tensor, "... dim"]
    logits: Float[torch.Tensor, "... c"] | None = None
    beta: float | None = None
    l: float | None = None
    r: float | None = None


def sae_hook(
    x: Float[torch.Tensor, "... dim"],
    hook: HookPoint | None,
    sae: SAE | BayesianSAE | torch.nn.Module,
    hook_acts: dict[str, Any],
    hook_key: str,
) -> Float[torch.Tensor, "... dim"]:
    """Runs the SAE on the input and stores the input, output and c in hook_acts under hook_key.

    If the SAE is a BayesianSAE, also stores the logits.

    Args:
        x: The input.
        hook: HookPoint object. Unused.
        sae: The SAE to run the input through.
        hook_acts: Dictionary of SAEActs and CacheActs objects to store the input, c, and output in.
        hook_key: The key in hook_acts to store the input, c, and output in.

    Returns:
        The output of the SAE.
    """
    if isinstance(sae, BayesianSAE):
        output, c, logits, beta, l, r = sae(x)
        hook_acts[hook_key] = SAEActs(input=x, c=c, output=output, logits=logits, beta=beta, l=l, r=r)
    elif isinstance(sae, SAE):
        output, c = sae(x)
        hook_acts[hook_key] = SAEActs(input=x, c=c, output=output, logits=None)
    else:
        # Fallback for generic torch.nn.Module - assumes (output, c) signature
        # Might need adjustment if other SAE types are used
        output, c = sae(x) # type: ignore
        hook_acts[hook_key] = SAEActs(input=x, c=c, output=output, logits=None)

    return output


def cache_hook(
    x: Float[torch.Tensor, "... dim"],
    hook: HookPoint | None,
    hook_acts: dict[str, Any],
    hook_key: str,
) -> Float[torch.Tensor, "... dim"]:
    """Stores the input in hook_acts under hook_key.

    Args:
        x: The input.
        hook: HookPoint object. Unused.
        hook_acts: CacheActs object to store the input in.

    Returns:
        The input.
    """
    hook_acts[hook_key] = CacheActs(input=x)
    return x
