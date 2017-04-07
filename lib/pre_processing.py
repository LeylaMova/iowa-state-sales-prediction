import pandas as pd
import numpy as np
import re


def money_toFloat(col):
    
    return col.str.replace('$', '').str.replace(',', '').astype(float)

    
def format_text(col):
    
    return col.str.lower().str.title()   

    
def clean_county(row):
    
    row = str(row).lower().title()
    row = np.where(row == 'Buena Vist', 'Buena Vista', row)
    row = np.where(row == 'Pottawatta', 'Pottawattamie', row)
    row = np.where(row == 'Obrien', "O'Brien", row)
    row = np.where(row == 'Cerro Gord', 'Cerro Gordo', row)
    
    return row   


def top_features(feature_names, coef_):
    df = pd.DataFrame(zip(feature_names,coef_), columns=['features','coefficients']) 
    
    return df.reindex(df.coefficients.abs().sort_values(ascending=False).index)[:10]


def score_me(scorer, model, X, y):
    return scorer(model.predict(X), y)















