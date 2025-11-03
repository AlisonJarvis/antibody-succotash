import pandas as pd
from feature_utils import create_features_from_raw_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def divide_into_train_test(csv_input_path, csv_target_path, feature_list, target_feature="HIC", test_size=0.2):

    ######################### Compile relevant dataframes ######################

    # Let's first get the derived features and the target from the inputs
    sequences = pd.read_csv(csv_input_path)
    properties = pd.read_csv(csv_target_path)
    # Get the derived properties
    sequence_features = create_features_from_raw_df(sequences)
    # Combine these into one dataframe with target and features
    features_and_predictions = pd.merge(sequence_features, properties[["antibody_id", target_feature]], left_on="antibody_id", right_on="antibody_id")
    # Remove nans
    features_and_predictions_clean = features_and_predictions.dropna()

    ########################## Divide into X and y ############################

    if feature_list == 'all':
        X = features_and_predictions_clean.drop(columns=['antibody_id', target_feature])
        y = features_and_predictions_clean[target_feature]
    else:
        X = features_and_predictions_clean[feature_list]
        y = features_and_predictions_clean[target_feature]

    ######################### Train test division #############################

    # Divide into train and test with scikit learn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Min max scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test