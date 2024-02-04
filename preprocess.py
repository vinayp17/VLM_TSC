import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.model_selection import train_test_split

def load_birds(path):
    raw_data = loadarff(path)

    df = pd.DataFrame(raw_data[0])

    birds = [i for i in df.columns if 'audio' not in i and 'cluster' not in i][9:]
    audio = [i for i in df.columns if 'audio' in i]
    for i in birds:
        df[i] = pd.to_numeric(df[i])
        
    # Function to get present variables for each row
    def get_present_variables(row):
        if len([col for col in df.columns if row[col] == 1 and col in birds]) > 1:
            output = ', '.join([col for col in df.columns if row[col] == 1 and col in birds])
            output = output.replace("\'", "").replace("\\", "")
            return output.lower()
        else:
            return "no bird"

    # Apply the function to each row
    df['Present_Variables'] = df.apply(get_present_variables, axis=1)
    vc = df['Present_Variables'].value_counts().reset_index()
    df = df[df['Present_Variables'].isin(vc[vc['count'] > 1]['Present_Variables'].values)]

    # X_train, X_test, y_train, y_test = train_test_split(df[audio], df['Present_Variables'], test_size=0.2, stratify=df['Present_Variables'], random_state=42)
    return df[audio].values, df['Present_Variables'].values