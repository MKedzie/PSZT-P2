import pandas as pd

def logistic_regression():
    plik=pd.read_csv('../Data/winequality-red.csv',header=0)
    data=plik.dropna()
    print(data.shape)
    print(list(data.columns))
    
