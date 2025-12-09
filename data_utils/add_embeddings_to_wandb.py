import wandb
from pathlib import Path
from embedding_utils import get_model_embeddings
import pandas as pd

def main():
    artifact = wandb.Artifact(name="p-IgGen_transformer_embeddings", type="dataset")
    parent_dir = Path(__file__).parent.absolute()
    out_path = parent_dir.joinpath("embeddings_sr.pkl")

    with wandb.init(project="Antibody Succotash") as run:
        # Use gdpa dataset
        gdpa_dataset = run.use_artifact("GDPa_Dataset:latest")
        csvs_loc = gdpa_dataset.download(parent_dir.parent.joinpath("wandb").as_posix())

        raw_df = pd.read_csv(Path(csvs_loc).joinpath("GDPa1_v1.2_20250814.csv"))
        embeddings_sr = get_model_embeddings(raw_df)
        embeddings_sr.to_pickle(out_path.as_posix())

        artifact.add_file(local_path=out_path.as_posix())
        run.log_artifact(artifact)
        print(parent_dir.absolute().as_posix())
        run.log_code(parent_dir.absolute().as_posix())

if __name__=="__main__":
    main()