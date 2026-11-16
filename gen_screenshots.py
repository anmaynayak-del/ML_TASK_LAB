import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

base = r"C:\Users\Anirudh\Downloads\ml2"

def save_text_as_image(text, title, out_path, bg="#1e1e1e", fg="#d4d4d4"):
    lines = text.strip().split('\n')
    n = len(lines)
    fig_height = max(4, n * 0.26 + 1.4)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis('off')

    # Title bar
    fig.text(0.03, 0.97, f"  {title}", fontsize=10.5, color='white',
             fontfamily='monospace', fontweight='bold', va='top',
             bbox=dict(facecolor='#3c3c3c', edgecolor='none', boxstyle='round,pad=0.3'))
    for xi, col in zip([0.88, 0.91, 0.94], ['#ff5f57','#febc2e','#28c840']):
        fig.text(xi, 0.97, '●', fontsize=13, color=col, va='top')

    ax.text(0.012, 0.97, text,
            transform=ax.transAxes,
            fontsize=8.5, color=fg,
            fontfamily='monospace',
            va='top', ha='left', linespacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=bg)
    plt.close()
    print(f"Saved: {out_path}")

# ---- Random Forest Output ----
rf_output = """Shape of dataset (rows, columns): (569, 33)

************** First 5 rows of dataset **************
         id diagnosis  ...  fractal_dimension_worst  Unnamed: 32
0    842302         M  ...                  0.11890          NaN
1    842517         M  ...                  0.08902          NaN
2  84300903         M  ...                  0.08758          NaN
3  84348301         M  ...                  0.17300          NaN
4  84358402         M  ...                  0.07678          NaN

[5 rows x 33 columns]

Features set:
   radius_mean  texture_mean  ...  symmetry_worst  fractal_dimension_worst
0        17.99         10.38  ...          0.4601                  0.11890
1        20.57         17.77  ...          0.2750                  0.08902
2        19.69         21.25  ...          0.3613                  0.08758
3        11.42         20.38  ...          0.6638                  0.17300
4        20.29         14.34  ...          0.2364                  0.07678

Target labels:
0    M
1    M
2    M
3    M
4    M
Name: diagnosis, dtype: str

=============================================
             ACCURACY SECTION
=============================================
Training Accuracy   : 0.9824 (98.24%)
Testing Accuracy    : 0.9561 (95.61%)
Correct Predictions : 109 / 114
Misclassified       : 5 / 114
=============================================

=== Confusion Matrix ===
[[69  2]
 [ 3 40]]

=== Classification Report ===
               precision    recall  f1-score   support

   Benign (B)       0.96      0.97      0.97        71
Malignant (M)       0.95      0.93      0.94        43

     accuracy                           0.96       114
    macro avg       0.96      0.95      0.95       114
 weighted avg       0.96      0.96      0.96       114

Plot saved as 'true_vs_predicted.png'"""

save_text_as_image(rf_output, "Random Forest Classifier — Output",
                   os.path.join(base, "rf_output.png"))

# ---- Backpropagation Output ----
bp_output = """==================================================
  PART 1: Backpropagation on XOR Problem
==================================================

Input:
[[0 0]
 [0 1]
 [1 0]
 [1 1]]

Actual Output:
[[0]
 [1]
 [1]
 [0]]

Predicted Output (raw probabilities):
[[0.12484698]
 [0.86256456]
 [0.86541684]
 [0.13389571]]

Predicted Output (thresholded at 0.5):
0
1
1
0

=============================================
   ACCURACY SECTION — XOR Problem
=============================================
Correct Predictions : 4 / 4
Accuracy            : 100.00%
=============================================

==================================================
  PART 2: Backpropagation on Iris Dataset
==================================================

Dataset shape: (150, 6)
Classes: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

Training for 5000 epochs...

=============================================
   ACCURACY SECTION — Iris Dataset
=============================================
Training Accuracy   : 98.00%
Correct Predictions : 147 / 150
Misclassified       : 3 / 150
---------------------------------------------
Per-Class Accuracy:
  Iris-setosa         : 100.00%  (50/50)
  Iris-versicolor     :  96.00%  (48/50)
  Iris-virginica      :  98.00%  (49/50)
============================================="""

save_text_as_image(bp_output, "Backpropagation Neural Network — Output",
                   os.path.join(base, "bp_output.png"))

print("All images saved!")
