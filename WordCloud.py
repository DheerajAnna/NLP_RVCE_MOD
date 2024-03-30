import matplotlib.pyplot as plt
import numpy as np

def plot_wordcloud(word_freq):
    # Get the maximum frequency
    max_freq = max(word_freq.values())

    # Create random positions for words
    np.random.seed(42)  # For reproducibility
    positions = np.random.rand(len(word_freq), 2)

    # Plot each word with its scaled frequency as font size and random color
    for (word, freq), (x, y) in zip(word_freq.items(), positions):
        scaled_freq = freq / max_freq
        color = np.random.rand(3,)  # Random RGB color
        plt.text(x, y, word, fontsize=100 * scaled_freq, color=color, ha='center', va='center')

    plt.axis('off')
    plt.show()

# Example usage
word_freq = {
    'apple': 10,
    'banana': 20,
    'orange': 5,
    'grape': 15
}

plot_wordcloud(word_freq)
