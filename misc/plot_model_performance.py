import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_model_performance(xlsx_loc='../misc_img/model scores.xlsx'):
    df = pd.read_excel(xlsx_loc)
    n = len(df)
    melted = pd.melt(df)
    fig = plt.figure(figsize=(16, 6))
    # runs and model
    fig.add_subplot(1, 4, 1)
    sns.boxplot(x=melted['variable'], y=melted['value'])
    plt.xlabel('Model')
    plt.ylabel('Distance travelled in-game(m)')
    plt.title(f'Distance travelled in different runs(n={n})', fontsize=10)

    fig.add_subplot(1, 4, 2)
    sns.scatterplot(x=melted['variable'], y=melted['value'])
    plt.xlabel('Model')
    plt.ylabel('Distance travelled in-game(m)')
    plt.title(f'Distance travelled in different runs(n={n})', fontsize=10)

    # logruns and model
    melted['value'] = np.log(melted['value'])
    fig.add_subplot(1, 4, 3)
    sns.boxplot(x=melted['variable'], y=melted['value'])
    plt.xlabel('Model')
    plt.ylabel('ln of distance travelled in-game')
    plt.title(f'ln Distance travelled in different runs(n={n})', fontsize=10)

    fig.add_subplot(1, 4, 4)
    sns.scatterplot(x=melted['variable'], y=melted['value'])
    plt.xlabel('Model')
    plt.ylabel('ln of distance travelled in-game')
    plt.title(f'ln Distance travelled in different runs(n={n})', fontsize=10)

    plt.show()


if __name__ == '__main__':
    plot_model_performance()
