import argparse

from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import Config
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import main as run_training
from e2e_sae.utils import load_config, replace_pydantic_model


def main(args: argparse.Namespace):
    base_config: Config = load_config(args.config, config_model=Config)
    updates = {
        "wandb_run_name_prefix": args.wandb_run_name_prefix
    }
    config = replace_pydantic_model(base_config, updates)
    run_training(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="e2e_sae/scripts/train_tlens_saes/tinystories_1M_local.yaml")
    parser.add_argument("--wandb_run_name_prefix", type=str, default="")
    args = parser.parse_args()
    main(args)