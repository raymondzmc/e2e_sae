from e2e_sae.utils import load_config, replace_pydantic_model
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_bayesian_saes import Config
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_bayesian_saes import main as run_bayesian_training
import argparse

def main(args: argparse.Namespace):
    base_config: Config = load_config(args.config, config_model=Config)
    updates = {
        "wandb_run_name_prefix": f"beta_{args.beta}_sparsity_coeff_{args.sparsity_coeff}_",
        "saes": {"hard_concrete_beta": args.beta},
        "loss": {
            "sparsity": {
                "beta": args.beta,
                "coeff": args.sparsity_coeff,
            },
        },
    }
    config = replace_pydantic_model(base_config, updates)
    run_bayesian_training(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml")
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--sparsity_coeff", type=float, default=0.5)
    args = parser.parse_args()
    for sparsity_coeff in [0.1, 0.2, 0.3, 0.4]:
        args.sparsity_coeff = sparsity_coeff
        main(args)