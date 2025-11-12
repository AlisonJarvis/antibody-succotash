import pandas as pd
import pathlib

from data_utils.embedding_utils import get_model_embeddings

def test_model_embeddings():
    csv_path = pathlib.Path(__file__).parent.parent.joinpath("data", "GDPa1_v1.2_sequences.csv")
    data_df = pd.read_csv(csv_path)[0:17]
    embeddings = get_model_embeddings(data_df)
    assert len(embeddings) == len(data_df)