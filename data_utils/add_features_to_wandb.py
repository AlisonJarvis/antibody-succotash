import wandb
from pathlib import Path
from feature_utils import create_features_from_raw_df
import pandas as pd

def main():
    artifact = wandb.Artifact(name="vh_vl_bioprot_features", type="dataset")
    parent_dir = Path(__file__).parent.absolute()
    out_path = parent_dir.joinpath("feature_df.pkl")

    with wandb.init(project="Antibody Succotash") as run:
        # Use gdpa dataset
        gdpa_dataset = run.use_artifact("GDPa_Dataset:latest")
        csvs_loc = gdpa_dataset.download(parent_dir.parent.joinpath("wandb").as_posix())

        raw_df = pd.read_csv(Path(csvs_loc).joinpath("GDPa1_v1.2_20250814.csv"))
        feature_df = create_features_from_raw_df(raw_df)
        feature_df.to_pickle(out_path.as_posix())

        artifact.add_file(local_path=out_path.as_posix())
        run.log_artifact(artifact)
        print(parent_dir.absolute().as_posix())
        run.log_code(parent_dir.absolute().as_posix())

if __name__=="__main__":
    main()