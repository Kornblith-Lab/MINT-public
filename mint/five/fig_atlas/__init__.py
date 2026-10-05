"""Figure: MINT Atlas of Pediatric ED Visits & CDR Optimization.

Panel A: UMAP of representations colored by HDBSCAN clusters
Panel B: ED-PEWS score vs hospital admission rate by cluster
Panel C: Description of adjusted ED-PEWS rule (text panel, drafted by user)
Panel D: AUROC/AUPRC curves before and after rule adjustment

Usage:
    python -m mint.five.fig_atlas
    python -m mint.five.fig_atlas --panel a
    python -m mint.five.fig_atlas --panel b
    python -m mint.five.fig_atlas --panel d
    python -m mint.five.fig_atlas --mode compute
    python -m mint.five.fig_atlas --mode plot
    python -m mint.five.fig_atlas --mode name
"""
