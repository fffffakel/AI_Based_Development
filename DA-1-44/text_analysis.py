import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_text_lengths(reviews, output_file):
    try:
        if not reviews or not isinstance(reviews, list):
            raise ValueError("Empty list")
        
        df = pd.DataFrame(reviews, columns=['review'])
        
        # length of each string
        df['length'] = df['review'].str.len()
        
        # longest and shortest string
        longest_review = df.loc[df['length'].idxmax()]
        shortest_review = df.loc[df['length'].idxmin()]
        
        # mean length
        mean_length = df['length'].mean()
        
        # histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['length'], bins=30, edgecolor='black')
        plt.title('Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        
        # Save
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        plt.savefig(output_path)
        plt.close()
        
        results = {
            'longest_review': {
                'text': longest_review['review'],
                'length': longest_review['length']
            },
            'shortest_review': {
                'text': shortest_review['review'],
                'length': shortest_review['length']
            },
            'mean_length': mean_length
        }
        
        return results
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    sample_reviews = [
        "Great product, really satisfied!",
        "Not bad, but could be better",
        "Amazing experience, highly recommend this to everyone!",
        "Okay product",
        "This is the best thing I've ever bought, fantastic quality and great service!",
        "Good item, fast delivery",
        "Could improve the quality",
        "Really happy with my purchase, will buy again!",
        "Good product"
    ]
    
    output_file = 'text_length_histogram.png'
    results = analyze_text_lengths(sample_reviews,output_file)
    
    if results:
        print("Analysis Results:")
        print(f"Longest string: ({results['longest_review']['length']}): {results['longest_review']['text']}")
        print(f"Shortest string ({results['shortest_review']['length']}): {results['shortest_review']['text']}")
        print(f"Mean Length: {results['mean_length']:.2f}")