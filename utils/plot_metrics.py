from typing import List
import matplotlib.pyplot as plt
def plot_insertion_score(rates: List[int], insertion_values_per_rate: List[int], insertion_auc_score:float, output_dir:str):
    # Plot graphic
    plt.figure(figsize=(8, 6))
    plt.plot(rates, insertion_values_per_rate, marker='o', linestyle='-', color='g', label=f'Grad-CAM (AUC = {insertion_auc_score:.4f})')
    plt.fill_between(rates, insertion_values_per_rate, alpha=0.2, color='g')

    plt.title('Insertion Score Curve', fontsize=14)
    plt.xlabel('Inserted pixels percentage', fontsize=12)
    plt.ylabel("Model's Confidence", fontsize=12)
    plt.xticks(rates, [f"{r*100:.0f}%" for r in rates])
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower right")
    # Save it and show
    plt.savefig(output_dir + "_"+ "auc_insertion_score.png", dpi=300)
    plt.show()
    

def plot_drop_in_confidence(rates: List[int], drop_values_per_rate: List[int], deletion_auc_score:float, output_dir: str):
    # Plot graphic
    plt.figure(figsize=(8, 6))

    plt.plot(rates, drop_values_per_rate, marker='o', linestyle='-', color='b', label=f'Grad-CAM (AUC = {deletion_auc_score:.4f})')
    plt.fill_between(rates, drop_values_per_rate, alpha=0.2, color='b')

    plt.title('Drop in Confidence Curve', fontsize=14)
    plt.xlabel('Deleted pixels percentage', fontsize=12)
    plt.ylabel('Averag Drop in Confidence', fontsize=12)
    plt.xticks(rates, [f"{r*100:.0f}%" for r in rates])
    plt.ylim(0, 1.05) 
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower right", fontsize=12)
    # Save it and show
    plt.savefig(output_dir + "_"+ "auc_drop_in_confidence.png", dpi=300)
    plt.show()