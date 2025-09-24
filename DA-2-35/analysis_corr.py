import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_collinearity(path):
    """
    Analyze collinearity in dataset by computing the correlation matrix
    and identifying feature pairs with high correlation (|corr| > 0.8).

    Parameters:
        path (str): File path to the CSV dataset.

    Returns:
        dict: A dictionary containing:
            - correlation_matrix: DataFrame of the correlation matrix.
            - high_corr_pairs: List of tuples containing feature pairs with |corr| > 0.8.
    """
    try:
        dataset = pd.read_csv(path)

        # Select only numeric columns for correlation analysis
        dataset = dataset.select_dtypes(include=[np.number])

        # Compute correlation matrix
        corr_matrix = dataset.corr()
        
        # Find pairs with |corr| > 0.8
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        results = {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
        }
        
        return results
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def visualize_correlation_matrix(corr_matrix, output_file):
    """
    Create and save a heatmap of the correlation matrix.

    Parameters:
        corr_matrix (DataFrame): Correlation matrix to visualize.
        output_file (str): File path to save the heatmap image.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='magma')
    plt.title('Correlation Matrix Heatmap')
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    output_file = 'correlation_matrix_heatmap.png'

    script_dir = os.path.dirname(__file__)
    path_to_dataset = os.path.join(script_dir, "Iris.csv")
    results = analyze_collinearity(path_to_dataset)
    
    if results:
        # Visualize the correlation matrix
        visualize_correlation_matrix(results['correlation_matrix'], output_file)
        
        # Print results
        print("Analysis Results:")
        print("\nFeature pairs with |correlation| > 0.8:")
        if results['high_corr_pairs']:
            for pair in results['high_corr_pairs']:
                print(f"{pair[0]} and {pair[1]}: {pair[2]:.3f}")
        else:
            print("No feature pairs with |correlation| > 0.8 found.")
    
    print()
    print("""Мультиколленеарность пробелема:
1.Нестабильность оценок модели - коэффициенты модели cтановятся нестабильными и чувствительными к небольшим изменениям в данных
2.Снижение интерпретируемости - когда признаки сильно коррелируют, сложно определить, какой из них действительно влияет на целевую переменную.
3.Переобучение и плохая обобщающая способность - высокая мультиколлинеарность может привести к переобучению модели
            """)


#Не взял boston, тк он удален из новых scikit-learn
# ImportError: 
# `load_boston` has been removed from scikit-learn since version 1.2.