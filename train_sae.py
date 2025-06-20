import argparse

from e2e_sae.scripts.train_tlens_saes.run_train_tlens_bayesian_saes import Config as BayesianConfig
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_bayesian_saes import (
    main as run_bayesian_training,
)
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import Config
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import main as run_training
from e2e_sae.utils import load_config, replace_pydantic_model


def main(args: argparse.Namespace):
    if args.bayesian_sae:
        config_model = BayesianConfig
        wandb_run_name_prefix = f"{args.wandb_run_name_prefix}_bayesian_"
    else:
        config_model = Config
        wandb_run_name_prefix = f"{args.wandb_run_name_prefix}_"

    base_config: Config | BayesianConfig = load_config(args.config, config_model=config_model)
    updates = {
        "wandb_run_name_prefix": wandb_run_name_prefix,
        "lr": args.lr,
        "loss": {
            "sparsity": {},
        },
        "saes": {},
    }
    if args.wandb_tags:
        updates["wandb_tags"] = args.wandb_tags
    if args.wandb_project:
        updates["wandb_project"] = args.wandb_project
    if args.bayesian_sae:
        updates["saes"]["initial_beta"] = args.initial_beta
        updates["saes"]["final_beta"] = args.final_beta
    if args.sparsity_coeff is not None:
        updates["loss"]["sparsity"]["coeff"] = args.sparsity_coeff
    if args.layer:
        updates["saes"]["sae_positions"] = [f"blocks.{args.layer}.hook_resid_pre"]

    config = replace_pydantic_model(base_config, updates)
    for seed in range(args.num_seeds):
        seed_config = replace_pydantic_model(config, {"seed": seed})
        if args.bayesian_sae:
            run_bayesian_training(seed_config)
        else:
            run_training(seed_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sparsity_coeff", type=float, default=None, help="Initial coefficient for sparsity loss.")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name_prefix", type=str, default=None)
    parser.add_argument("--bayesian_sae", action="store_true")
    parser.add_argument("--wandb_tags", nargs='+', type=str, default=None, help="Tags to add to the wandb run.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initial_beta", type=float, default=1.0)
    parser.add_argument("--final_beta", type=float, default=0.1)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()
    main(args)