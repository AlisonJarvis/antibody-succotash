from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def get_model_embeddings(sequence_df: pd.DataFrame) -> pd.Series:
    '''
    Given an input `DataFrame` with `"vh_protein_sequence"` and `"vl_protein_sequence"` columns,
    generates an embedding for each token using the p-IgGen transformer model. Returns results as
    a list of numpy arrays.

    Args:
        sequence_df (DataFrame): The input dataframe containing the vh and vl protein sequences.

    Returns:
        Series: A series with elements consisting of token level embeddings of each sequence.
    '''
    model_name = "ollieturnbull/p-IgGen"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Paired sequence handling: Concatenate heavy and light chains and add beginning ("1") and end ("2") tokens
    # (e.g. ["EVQLV...", "DIQMT..."] -> "1E V Q L V ... D I Q M T ... 2")
    sequences = [
        "1" + " ".join(heavy) + " ".join(light) + "2"
        for heavy, light in zip(
            sequence_df["vh_protein_sequence"],
            sequence_df["vl_protein_sequence"],
        )
    ]

    # Load model
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    batch_size = 16
    full_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = tokenizer(sequences[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
        outputs = model(batch["input_ids"].to(device), return_rep_layers=[-1], output_hidden_states=True)
        embeddings = outputs["hidden_states"][-1].detach().cpu().numpy()
        for i in range(embeddings.shape[0]):
            full_embeddings.append(embeddings[i])
    return pd.Series(full_embeddings, index=sequence_df.index)