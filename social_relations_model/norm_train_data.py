import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def norm_train():
    df = pd.read_csv("data/model_data.csv")
    columns = ["dist", "speed", "angle"]

    for feature in columns:
        df[feature] = MinMaxScaler().fit_transform(np.array(df[feature]).reshape(-1,1))
    
    df.to_csv("data/model_data_normalized.csv", index=False)

def norm_per_frame():
    df = pd.read_csv("data/model_data_01_group_labels.csv")
    columns = ["dist", "speed", "angle"]
    frames = df.frameID.unique()

    df_out = pd.DataFrame()
    for frame in frames:
        df_current = df.loc[df.frameID == frame]
        for feature in columns:
            df_current[feature] = MinMaxScaler().fit_transform(np.array(df_current[feature]).reshape(-1,1))
        df_out = pd.concat([df_out, df_current])
        
    
    df.to_csv("data/model_data_norm_per_frame_01_group_labels.csv", index=False)

def format_group_labels():
    df = pd.read_csv("data/model_data.csv")
    columns = ['group_label']

    x = df[['group_label']].values.astype(int)
    x_scaled = MinMaxScaler().fit_transform(x).astype(int)
    df['group_label'] = pd.DataFrame(x_scaled)
    
    df.to_csv("data/model_data_01_group_labels.csv", index=False)

def main():
    #format_group_labels()
    norm_per_frame()

if __name__ == "__main__":
    main()