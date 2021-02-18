import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    df = pd.read_csv("data/model_data.csv")
    columns = ["dist", "speed", "angle"]

    for feature in columns:
        df[feature] = MinMaxScaler().fit_transform(np.array(df[feature]).reshape(-1,1))
    
    df.to_csv("data/model_data_normalized.csv", index=False)

if __name__ == "__main__":
    main()