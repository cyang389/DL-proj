import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
from torch_geometric.data import Data


np.random.seed(12345)

initialize = True

def process():
    if not initialize:

        df = pd.read_csv('./yoochoose-data/yoochoose-clicks.dat', header=None)
        df.columns=['session_id','timestamp','item_id','category']
        sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
        df = df.loc[df.session_id.isin(sampled_session_id)]
        buy_df = pd.read_csv('./yoochoose-data/yoochoose-buys.dat', header=None)
        buy_df.columns=['session_id','timestamp','item_id','price','quantity']
        df['label'] = df.session_id.isin(buy_df.session_id)

        pickle.dump(df, open("./yoochoose-data/yoochoose-clicks.p", "wb"))
        pickle.dump(df, open("./yoochoose-data/yoochoose-buys.p", "wb"))
    else:
        df = pickle.load(open("./yoochoose-data/yoochoose-clicks.p", "rb"))
        buy_df = pickle.load(open("./yoochoose-data/yoochoose-buys.p", "rb"))

    return df