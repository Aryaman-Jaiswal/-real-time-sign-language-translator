# ==============================================================================
# plot_history.py
#
# Description:
# This script reads the training history saved from our Kaggle notebook
# and uses Matplotlib to generate and save the accuracy and loss graphs
# required for the project report.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
HISTORY_CSV_PATH = 'srn_training_history.csv'
ACCURACY_PLOT_PATH = 'srn_accuracy_plot.png'
LOSS_PLOT_PATH = 'srn_loss_plot.png'

def generate_plots():
    """Loads the history CSV and generates the plots."""
    try:
        df = pd.read_csv(HISTORY_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{HISTORY_CSV_PATH}' was not found.")
        print("Please make sure you have downloaded it from your Kaggle notebook.")
        return

    # Get the number of epochs the model actually ran for
    num_epochs = len(df)
    epochs_range = range(1, num_epochs + 1)

    # --- 1. Generate the Accuracy Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, df['accuracy'], label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs_range, df['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
    
    plt.title('SRN Model Accuracy Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1.05) # Set y-axis from 0 to 105%
    
    # Save the plot to a file
    plt.savefig(ACCURACY_PLOT_PATH)
    print(f"Accuracy plot saved to: {ACCURACY_PLOT_PATH}")

    # --- 2. Generate the Loss Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, df['loss'], label='Training Loss', color='blue', marker='o')
    plt.plot(epochs_range, df['val_loss'], label='Validation Loss', color='orange', marker='o')
    
    plt.title('SRN Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Loss plot saved to: {LOSS_PLOT_PATH}")

    # Optional: Display the plots on screen
    # plt.show()

if __name__ == '__main__':
    generate_plots()