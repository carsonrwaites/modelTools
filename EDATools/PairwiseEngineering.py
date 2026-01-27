import numpy as np
import math
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px



def engineer(df, target, categorical=None, log=None, exponential=None, interaction=None, power=None, loud=False):
    # Log transform
    if log is not None:
        for term in log:
            df['ln('+term+')'] = df[term].apply(lambda x: np.log(x))
            df.drop(term, axis=1, inplace=True)
    # Exponential transform
    if exponential is not None:
        for term in exponential:
            df['exp('+term+')'] = df[term].apply(lambda x: np.exp(x))
            df.drop(term, axis=1, inplace=True)
    # Power terms
    if power is not None:
        for term in power.keys():
            nums = [i for i in range(2, power.get(term)+1)]
            labels = [term+f"^{i}" for i in nums]
            for indx, label in enumerate(labels):
                df[label] = df[term].apply(lambda x: np.power(x, nums[indx]))
    # Categorical terms
    if categorical is not None:
        for term in categorical:
            dummies = pd.get_dummies(df[term],
                                     drop_first=True,
                                     dtype=int,
                                     prefix=term)
            df = df.drop(term, axis=1)
            df = pd.concat([df, dummies], axis=1)
    # Interaction terms
    if interaction is not None:
        for pair in interaction:
            new_term = f"{pair[0]}*{pair[1]}"
            df[new_term] = df[pair[0]] * df[pair[1]]
    if loud:
        plot_pairwise(df, target)
    return df


def plot_pairwise(df, target):
    cols = df.columns.tolist()
    cols.remove(target)
    num_rows = math.ceil(len(cols) / 3)

    indx = 0
    fig = make_subplots(rows=num_rows, cols=3, shared_yaxes=True, subplot_titles=cols)
    for i in range(num_rows):
        for j in range(3):
            if indx <= len(cols) - 1:
                fig_px = px.scatter(df, x=cols[indx], y=target, trendline='ols')
                for trace in fig_px.data:
                    fig.add_trace(trace, row=i+1, col=j+1)
                indx += 1
    fig.update_layout(autosize=True, height = 3000)
    fig.show()
    return None

if __name__ == '__main__':
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=165)
    X = dataset.data.features
    y = dataset.data.targets
    df = X.copy()
    df['y'] = y
    df['dummy'] = np.random.randint(0, 3, len(df))

    df = engineer(df, 'y',
                  #log=['Age'],
                  categorical=['dummy'],
                  interaction=[('Age', 'Cement'), ('Cement', 'Fly Ash')],
                  loud=True)
    print(df.head())