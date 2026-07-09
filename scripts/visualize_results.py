"""Generate beautiful charts to visualize PaddleOCR evaluation results.

Reads results from paddleocr_archive_eval_v2.csv and saves charts (PNGs)
to the results/archive/ directory.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    # Attempt to import visualization libraries, prompting install if missing
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Error: matplotlib and seaborn are required for visualization.")
        print("Please run: pip install matplotlib seaborn")
        sys.exit(1)

    eval_csv = project_root / "results" / "archive" / "paddleocr_archive_eval_v2.csv"
    if not eval_csv.exists():
        print(f"Error: Evaluation file not found at {eval_csv}")
        sys.exit(1)

    print(f"Reading evaluation data from {eval_csv}...")
    df = pd.read_csv(eval_csv)

    # Output directory for charts
    output_dir = project_root / "results" / "archive"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # --- CHART 1: Raw vs Aligned CER Comparison (Log/Truncated Scale) ---
    print("Generating Chart 1: CER Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We clip raw CER for visualization because it goes up to 100+ (10000%)
    raw_cer_clipped = np.clip(df["cer"], 0, 5.0)  # Clip at 500% error
    aligned_cer = df["cer_aligned"] * 100  # Convert aligned to percentage (0-100%)

    # Plot aligned CER distribution
    sns.histplot(
        aligned_cer,
        bins=30,
        kde=True,
        color="#2a9d8f",
        label="Segment-Aligned CER (Normalized)",
        alpha=0.7,
        ax=ax
    )
    
    ax.set_title("Distribution of Segment-Aligned Character Error Rate (CER)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Character Error Rate (CER %)", fontsize=12)
    ax.set_ylabel("Number of Documents", fontsize=12)
    ax.set_xlim(0, 100)
    
    # Add mean line
    mean_val = aligned_cer.mean()
    ax.axvline(mean_val, color="#e76f51", linestyle="--", linewidth=2, label=f"Mean CER: {mean_val:.2f}%")
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    chart1_path = output_dir / "cer_aligned_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f" Saved: {chart1_path}")

    # --- CHART 2: Performance by Document Category ---
    if "category" in df.columns and len(df["category"].unique()) > 1:
        print("Generating Chart 2: Category Performance...")
        # Group by category and compute mean metrics
        cat_stats = df.groupby("category").agg(
            avg_cer_aligned=("cer_aligned", lambda x: x.mean() * 100),
            avg_wer_aligned=("wer_aligned", lambda x: x.mean() * 100),
            count=("document_id", "count")
        ).reset_index()

        # Sort by best performing (lowest CER)
        cat_stats = cat_stats.sort_values(by="avg_cer_aligned")

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Melt the dataframe for side-by-side bar chart
        melted = pd.melt(
            cat_stats, 
            id_vars=["category"], 
            value_vars=["avg_cer_aligned", "avg_wer_aligned"],
            var_name="Metric", 
            value_name="Percentage"
        )
        
        melted["Metric"] = melted["Metric"].map({
            "avg_cer_aligned": "Avg CER (Char Error %)",
            "avg_wer_aligned": "Avg WER (Word Error %)"
        })

        sns.barplot(
            data=melted,
            y="category",
            x="Percentage",
            hue="Metric",
            palette=["#2a9d8f", "#f4a261"],
            ax=ax
        )

        ax.set_title("Average Error Rates (CER & WER) by Document Category", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Error Rate (%) - Lower is Better", fontsize=12)
        ax.set_ylabel("Document Category", fontsize=12)
        ax.set_xlim(0, 105)
        ax.legend(fontsize=11, loc="lower right")

        # Add text labels on the bars for precision
        for p in ax.patches:
            width = p.get_width()
            if width > 0:
                ax.text(
                    width + 1.5,
                    p.get_y() + p.get_height() / 2,
                    f"{width:.1f}%",
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="semibold",
                    color="#333333"
                )

        plt.tight_layout()
        chart2_path = output_dir / "performance_by_category.png"
        plt.savefig(chart2_path, dpi=300)
        plt.close()
        print(f" Saved: {chart2_path}")

    # --- CHART 3: Confidence vs Aligned CER Scatter Plot ---
    if "avg_confidence" in df.columns:
        print("Generating Chart 3a: Confidence Distribution...")
        valid_data = df.dropna(subset=["avg_confidence", "cer_aligned"]).copy()
        
        # --- Chart 3a: Confidence Distribution ---
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            valid_data["avg_confidence"],
            bins=40,
            kde=True,
            color="#457b9d",
            ax=ax
        )
        ax.set_title("Distribution des Scores de Confiance (Modèle OCR)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Confiance du Modèle (%)", fontsize=12)
        ax.set_ylabel("Nombre de documents", fontsize=12)
        ax.set_xlim(0, 105)
        
        plt.tight_layout()
        chart3a_path = output_dir / "confidence_distribution.png"
        plt.savefig(chart3a_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {chart3a_path}")

        # --- Chart 3b: Boxplot of CER by Confidence Bins ---
        print("Generating Chart 3b: Boxplot CER by Confidence Bins...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bins for confidence
        bins = [0, 60, 70, 80, 90, 100]
        labels = ["< 60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        valid_data["conf_bin"] = pd.cut(valid_data["avg_confidence"], bins=bins, labels=labels, right=True)
        
        # Filter extreme CER for a clean boxplot and scale to %
        valid_data_capped = valid_data[valid_data["cer_aligned"] <= 1.1].copy()
        valid_data_capped["cer_pct"] = valid_data_capped["cer_aligned"] * 100
        
        # Plot the boxplot (showfliers=False prevents drawing thousands of outlier dots)
        sns.boxplot(
            x="conf_bin",
            y="cer_pct",
            data=valid_data_capped,
            palette="Blues",
            showfliers=False,
            width=0.6,
            ax=ax
        )
        
        # Add a light stripplot on top to show the actual bimodal clusters (0 and 100)
        sns.stripplot(
            x="conf_bin",
            y="cer_pct",
            data=valid_data_capped,
            color="#e76f51",
            alpha=0.15,
            size=3,
            jitter=0.25,
            ax=ax
        )
        
        ax.set_title("Taux d'Erreur (CER) par Intervalle de Confiance", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Intervalle de Confiance", fontsize=12)
        ax.set_ylabel("Character Error Rate (CER %)", fontsize=12)
        ax.set_ylim(-5, 115)
        
        plt.tight_layout()
        chart3b_path = output_dir / "cer_by_confidence_boxplot.png"
        plt.savefig(chart3b_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {chart3b_path}")

    print("\n🎉 Visualizations generated successfully in results/archive/!")


if __name__ == "__main__":
    main()
