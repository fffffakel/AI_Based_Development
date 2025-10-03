import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def analyze(path_to_dataset):
    """
    Apply t-SNE and PCA to reduce the Iris dataset to 2D and visualize the results.

    Returns:
        dict: A dictionary containing:
            - tsne_data: 2D array of t-SNE transformed data.
            - pca_data: 2D array of PCA transformed data.
            - target: Array of target labels.
            - target_names: List of target class names.
    """
    try:
        if not os.path.exists(path_to_dataset):
            raise FileNotFoundError(f"The file {path_to_dataset} does not exist.")
        
        # Load Iris dataset
        dataset = pd.read_csv(path_to_dataset)
        X = dataset.drop(columns="Species")
        y = dataset["Species"]
        target_names = list(dataset["Species"].unique())

        # Convert string labels to numerical indices
        y_numeric = pd.Categorical(y, categories=target_names).codes

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(X_scaled)

        #PCA
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(X_scaled)

        results = {
            'tsne_data': tsne_data,
            'pca_data': pca_data,
            'target': y_numeric,
            'target_names': target_names,
        }

        return results

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def visualize(tsne_data, pca_data, target, target_names, output_tsne, output_pca):
    """
    Create and save scatter plots for t-SNE and PCA 2D projections, colored by target variable.

    Parameters:
        tsne_data (array): 2D array of t-SNE transformed data.
        pca_data (array): 2D array of PCA transformed data.
        target (array): Array of target labels.
        target_names (list): List of target class names.
        output_tsne (str): File path to save the t-SNE scatter plot.
        output_pca (str): File path to save the PCA scatter plot.

    Returns:
        None
    """
    # Unique colors for each class
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_names)))

    # t-SNE scatter plot
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(target_names):
        idx = target == i
        plt.scatter(tsne_data[idx, 0], tsne_data[idx, 1], c=[colors[i]], label=class_name, s=50)
    plt.legend(title="Classes", loc='best')
    output_path_tsne = os.path.join(os.path.dirname(__file__), output_tsne)
    plt.savefig(output_path_tsne)
    plt.close()
    print(f"t-SNE scatter plot saved as {output_path_tsne}")

    # PCA scatter plot
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(target_names):
        idx = target == i
        plt.scatter(pca_data[idx, 0], pca_data[idx, 1], c=[colors[i]], label=class_name, s=50)
    plt.legend(title="Classes", loc='best')
    output_path_pca = os.path.join(os.path.dirname(__file__), output_pca)
    plt.savefig(output_path_pca)
    plt.close()
    print(f"PCA scatter plot saved as {output_path_pca}")

if __name__ == "__main__":
    output_tsne = 'tsne.png'
    output_pca = 'pca.png'

    script_dir = os.path.dirname(__file__)
    path_to_dataset = os.path.join(script_dir, "Iris.csv")
    results = analyze(path_to_dataset)

    if results:
        # Visualize the results
        visualize(
            results['tsne_data'],
            results['pca_data'],
            results['target'],
            results['target_names'],
            output_tsne,
            output_pca
        )
    else:
        print("Failed. Check errors")

print()
print("""t-SNE лучше подходит для визуализации кластеров, особенно когда нужно выделить локальные структуры данных, но он медленнее и менее интерпретируем. PCA быстрее, более интерпретируем и лучше сохраняет глобальную структуру, но может быть менее эффективен для разделения сложных кластеров.""")