import pandas as pd

def main():
    df = pd.read_csv('data/group_track_data.csv')
    df_group = pd.read_csv('data/detected_groups3.csv')

    segment_nums = []
    datasets = []
    for _, row in df_group.iterrows():
        frame = row.frameID
        seg_num = df[df.frameID == frame].iloc[0].segment_num
        dataset = df[df.frameID == frame].iloc[0].dataset

        segment_nums.append(seg_num)
        datasets.append(dataset)

    df_group['segment_num'] = segment_nums
    df_group['dataset'] = datasets
    df_group.to_csv('data/detected_groups3_datasets.csv')

if __name__ == "__main__":
    main()