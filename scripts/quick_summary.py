import pandas as pd

df = pd.read_csv("results/archive/paddleocr_archive_eval_v2.csv")

print("=" * 55)
print("  RESUME APRES CORRECTION BIDI")
print("=" * 55)
print(f"  Documents evalues      : {len(df)}")
print("-" * 55)
print("  CER brut moyen         :", round(df['cer'].mean() * 100, 2), "%")
print("  WER brut moyen         :", round(df['wer'].mean() * 100, 2), "%")
print("-" * 55)
print("  CER normalise moyen    :", round(df['cer_normalized'].mean() * 100, 2), "%")
print("  WER normalise moyen    :", round(df['wer_normalized'].mean() * 100, 2), "%")
print("-" * 55)
print("  CER aligne moyen       :", round(df['cer_aligned'].mean() * 100, 2), "%")
print("  WER aligne moyen       :", round(df['wer_aligned'].mean() * 100, 2), "%")
print("-" * 55)
print("  Docs parfaits (CER=0)  :", (df['cer_aligned'] == 0).sum(), "sur", len(df))
print("  Docs < 10% CER aligne  :", (df['cer_aligned'] < 0.1).sum(), "sur", len(df))
print("  Docs < 30% CER aligne  :", (df['cer_aligned'] < 0.3).sum(), "sur", len(df))
print("=" * 55)

if "category" in df.columns:
    print("\n  PERFORMANCE PAR CATEGORIE (CER aligne moyen) :")
    print("-" * 55)
    cat = df.groupby("category")["cer_aligned"].mean().sort_values()
    for name, val in cat.items():
        bar = "█" * int(val * 20)
        print(f"  {name[:28]:<28} {val*100:5.1f}% {bar}")
    print("=" * 55)
