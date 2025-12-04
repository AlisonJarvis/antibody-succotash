import wandb

# Add provided csv files to wand
artifact = wandb.Artifact(name="GDPa_Dataset", type="dataset")
artifact.add_file("GDPa1_v1.2_20250814.csv")
artifact.add_file("GDPa1_v1.2_sequences.csv")

# Save to Wanddb
artifact.save("Antibody Succotash")
