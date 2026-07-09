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
        print("Generating Chart 3: Confidence vs CER...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Hexbin or scatter with alpha for dense plots
        hb = ax.hexbin(
            df["avg_confidence"],
            df["cer_aligned"] * 100,
            gridsize=30,
            cmap="YlGnBu",
            mincnt=1
        )
        
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Number of Documents", fontsize=11)
        
        # Add a trend line (simple linear fit)
        # Filter NaNs just in case
        valid_data = df.dropna(subset=["avg_confidence", "cer_aligned"])
        x = valid_data["avg_confidence"]
        y = valid_data["cer_aligned"] * 100
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m*x + b, color="#e76f51", linestyle="-", linewidth=2.5, label="Trendline")

        ax.set_title("OCR Confidence Score vs. Actual CER", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Model Confidence Score (%)", fontsize=12)
        ax.set_ylabel("Character Error Rate (CER %)", fontsize=12)
        ax.set_ylim(-5, 105)
        ax.set_xlim(df["avg_confidence"].min() - 5, 105)
        ax.legend(fontsize=11)

        plt.tight_layout()
        chart3_path = output_dir / "confidence_vs_cer.png"
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        print(f" Saved: {chart3_path}")

    print("\n🎉 Visualizations generated successfully in results/archive/!")


if __name__ == "__main__":
    main()
