import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import os

def capture_output(script_path, cwd):
    result = subprocess.run(
        ["python", "-u", script_path],
        capture_output=True, text=True, cwd=cwd
    )
    return result.stdout

def save_output_as_image(text, title, out_path, bg="#1e1e1e", fg="#d4d4d4", accent="#569cd6"):
    lines = text.strip().split('\n')
    n = len(lines)
    fig_height = max(4, n * 0.22 + 1.2)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis('off')

    # Title bar
    fig.text(0.03, 0.97, f"  {title}", fontsize=11, color='white',
             fontfamily='monospace', fontweight='bold', va='top',
             bbox=dict(facecolor='#3c3c3c', edgecolor='none', boxstyle='round,pad=0.3'))

    # Terminal dots
    for xi, col in zip([0.88, 0.91, 0.94], ['#ff5f57','#febc2e','#28c840']):
        fig.text(xi, 0.97, '●', fontsize=13, color=col, va='top')

    # Output text
    full_text = '\n'.join(lines)
    ax.text(0.01, 0.98, full_text,
            transform=ax.transAxes,
            fontsize=8.5,
            color=fg,
            fontfamily='monospace',
            va='top', ha='left',
            linespacing=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=bg)
    plt.close()
    print(f"Saved: {out_path}")

base = r"C:\Users\Anirudh\Downloads\ml2"

# --- Random Forest ---
rf_output = capture_output("random_forest.py", os.path.join(base, "RandomForest"))
save_output_as_image(rf_output, "Random Forest Classifier — Output",
                     os.path.join(base, "rf_output.png"))

# --- Backpropagation ---
bp_output = capture_output("backpropagation.py", os.path.join(base, "Backpropagation"))
save_output_as_image(bp_output, "Backpropagation Neural Network — Output",
                     os.path.join(base, "bp_output.png"))

print("Done!")
