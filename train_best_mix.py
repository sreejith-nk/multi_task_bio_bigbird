import dotenv
import hydra
import json
from itertools import combinations
from omegaconf import DictConfig
import pandas as pd

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

def load_dict_from_json_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)  # Load JSON data from file into a dictionary
            return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}

file_path = 'best_mix.json'
best_mix = load_dict_from_json_file(file_path)

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

    final_metrics={}
    for ds,mix in best_mix.items():
        
        data =ds + ","
        for d in mix:
            data += d + ","

        data = data[:-1]
        config.datamodule.dataset_name=data
        config.name = ds

        metrics = train(config)
        final_metrics[ds] = metrics

    result_df = pd.DataFrame(final_metrics)
    result_df.to_csv("results.csv")

if __name__ == "__main__":
    main()
