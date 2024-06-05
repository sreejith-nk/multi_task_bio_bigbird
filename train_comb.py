import dotenv
import hydra
from itertools import combinations
from omegaconf import DictConfig
import pandas as pd

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    data=config.datamodule.dataset_name.split(",")
    data_combs = list(combinations(data, 2))
    final_metrics={}

    for ds in data_combs:
        config.datamodule.dataset_name=ds[0]+","+ds[1]
        config.name = f"{config.datamodule.dataset_name}"
        metrics = train(config)
        final_metrics[ds[0]+"_"+ds[1]] = metrics
    
    for ds in data:
        config.datamodule.dataset_name=ds
        config.name = f"{config.datamodule.dataset_name}"
        metrics = train(config)
        final_metrics[ds] = metrics
        
    result_df = pd.DataFrame(final_metrics)
    result_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
