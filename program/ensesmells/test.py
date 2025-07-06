import gc

import torch
import numpy as np
import pandas as pd

pklFile = open("D:/codeSmell/JSS-EnseSmells/EnseSmells/program/ensesmells/DataClass_TokenIndexing_metrics.pkl", 'rb')
df = pd.read_pickle(pklFile)
print(df.head())