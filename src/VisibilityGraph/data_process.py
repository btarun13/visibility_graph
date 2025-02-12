import pandas as pd

def make_df_list(df, window, k,col_types:list):
    data_x = []
    label_list = []
    # Calculate the features
    for t in range(len(df)):
        future_index = t + window + k
        observed_space_end = t + window
        
        # Stop if we reach the end of the DataFrame (can't calculate future price)
        if future_index >= len(df) - 1:
            break
        
        X = df.iloc[t:t + window,:][col_types]
        data_x.append(X.reset_index(drop=True))
        del X

    # Calculate the labels
    for i in range(0,len(data_x)):
        
        if i == len(data_x)-1:
            break
        
        m = list(data_x[i]['close'])[window-1] - list(data_x[i+1]['close'])[k]
        if m > 0:
            label_list.append(1)
        elif m <= 0:
            label_list.append(0)
        del m

    data_x = data_x[:len(label_list)]
    return data_x, label_list

### data_x list and label list
def make_train_valid(data_x:list,data_y:list,tr_portion:float):
    train_set_x = data_x[:int(len(data_x) * tr_portion)]
    train_set_y = data_y[:int(len(data_y) * tr_portion)]

    valid_set_x = data_x[int(len(data_x) * tr_portion):]
    valid_set_y = data_y[int(len(data_y) * tr_portion):]

    return train_set_x, train_set_y, valid_set_x, valid_set_y

