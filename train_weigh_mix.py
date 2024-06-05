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

def get_entity_id(data_name):
    if data_name in ["mirna-disease","biored-disease","NCBI-disease", "BC5CDR-disease", "mirna-di", "ncbi_disease", "scai_disease", "variome-di"]:
        entity_id = 1
    elif data_name in ["iepa-chemical","biored-chemical","BC5CDR-chemical","cdr-ch", "chemdner", "scai_chemicals", "chebi-ch", "BC4CHEMD","drAbreu/bc4chemd_ner","bc4chemd_ner"]:
        entity_id = 2
    elif data_name in ["biored-gene","genetag","BC2GM","bc2gm","bc2gm_corpus","mirna-gp", "cell_finder-gp", "chebi-gp", "loctext-gp", "deca", "fsu", "progene", "jnlpba-gp", "bio_infer-gp", "variome-gp", "osiris-gp",  "iepa"]:
        entity_id = 3
    elif data_name in ["mirna-species","s800", "linnaeus", "loctext-sp", "mirna-sp", "chebi-sp", "cell_finder-sp", "variome-sp","species_800"]:
        entity_id = 4
    elif data_name in ["JNLPBA-cl", "cell_finder-cl", "jnlpba-cl", "gellus", "cll","biored-cl"]:
        entity_id = 5
    elif data_name in ["JNLPBA-dna", "jnlpba-dna"]:
        entity_id = 6
    elif data_name in ["JNLPBA-rna","jnlpba-rna"]:
        entity_id = 7
    elif data_name in ["JNLPBA-ct","jnlpba-ct"]:
        entity_id = 8
    elif data_name in ["mirna-protein","JNLPBA-protein","bioinfer-protein"]:
        entity_id = 9
    elif data_name in ["ddi_ner"]:
        entity_id = 10
    elif data_name in ["chem_prot"]:
        entity_id = 11
    elif data_name in ["ddi"]:
        entity_id = 12
    elif data_name in ["gad"]:
        entity_id = 13
    else:
        entity_id=0
    return entity_id

file_path = 'best_mix_copy.json'
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

        weight_index = get_entity_id(ds)
        config.model.weight_index = weight_index

        data = data[:-1]
        config.datamodule.dataset_name=data
        config.name = ds

        metrics = train(config)
        final_metrics[ds] = metrics

    result_df = pd.DataFrame(final_metrics)
    result_df.to_csv("results.csv")

if __name__ == "__main__":
    main()
