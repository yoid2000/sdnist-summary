from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, precision_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pprint
import json

pp = pprint.PrettyPrinter(indent=4)

catCols = ['SEX', 'HISP', 'EDU', 'OWN_RENT', 'RAC1P', 'INDP_CAT', 'MSP', 'PUMA',
'INDP', 'PINCP', 'WGTP', 'DPHY', 'DEYE', 'DEAR', 'NOC', 'PWGTP', 'DVET', 'NPF', 'POVPIP', 'HOUSING_TYPE', 'PINCP_DECILE', 'DREM', 'DENSITY']
numCols = ['AGEP']
quasi_identifiers = ['SEX', 'HISP', 'EDU', 'OWN_RENT', 'RAC1P', 'INDP_CAT', 'MSP', 'PUMA']
targets = ['INDP', 'PINCP', 'WGTP', 'DPHY', 'DEYE', 'DEAR', 'NOC', 'PWGTP', 'DVET', 'DENSITY', 'NPF', 'POVPIP', 'HOUSING_TYPE', 'PINCP_DECILE', 'AGEP', 'DREM',]

def getPrecisionFromBestGuess(y_test, dfCol):
    # Find the most frequent category from the source data
    most_frequent = dfCol.mode()[0]
    print(f"most_frequent is {most_frequent}")
    # Emulate precision if we had simply always predected this among the test data
    most_frequent_count = (y_test == most_frequent).sum()
    print(f"most_frequent_count {most_frequent_count}")
    return(most_frequent_count / len(y_test))

def convert_to_numpy(var):
    if isinstance(var, pd.Series):
        return var.values
    elif isinstance(var, np.ndarray):
        return var
    else:
        print("The input is neither a pandas Series nor a numpy array.")
        return None

def runModel(X, y, nums, cats, targetType, max_iter=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Create a column transformer
    print(f"cats = {cats}")
    print(f"nums = {nums}")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), nums),
            ('cat', OneHotEncoder(), cats)
        ])

    # Create a pipeline that uses the transformer and then fits the model
    if targetType == 'cat':
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', LogisticRegression(penalty='l1', C=0.01, solver='saga', max_iter=max_iter))])
    else:
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', Lasso(alpha=0.1))])

    # Fit the pipeline to the training data
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy = {accuracy}")
    return accuracy

if __name__ == "__main__":
    df_all = pd.read_csv('national2019.csv')
    # We just treat all columns as categorical
    df_all = df_all.astype('string')
    baselines = {}
    for target in targets:
        test_cols = quasi_identifiers + [target]
        cats = quasi_identifiers.copy()
        nums = []
        targetType = 'cat'
        df = df_all[test_cols]
        X = df.drop(target, axis=1)
        y = df[target]
        print(f"Test columns {X.columns}")
        print(f"Target column {target}")
        print(y.value_counts())
        accuracy = runModel(X, y, nums, cats, targetType)
        baselines[target] = accuracy
        with open('baselines.json', 'w') as f:
            json.dump(baselines, f, indent=4)