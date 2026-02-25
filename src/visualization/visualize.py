import seaborn as sns
import matplotlib.pyplot as plt

def histogram(df, column, save_path=None):
    """
    Affiche un histogramme pour une colonne donnée d'un DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à visualiser.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column])
    plt.title(f'Histogramme de {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f'histogram_{column}.png')

    plt.show()