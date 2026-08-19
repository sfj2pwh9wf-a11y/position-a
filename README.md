# Adikus Backgammon Analyzer — Full Position Build

This Streamlit app is designed for Adikus iPhone screenshots.

## Pipeline

1. Read the fresh screenshot.
2. Detect White and Black checkers on all 24 points.
3. Detect checkers on the bar.
4. Infer borne-off checkers from the 15-checker total.
5. Show the reconstructed White/Black position for verification.
6. Send the complete 2×25 board to `gnubg-nn`.
7. Analyze the current dice with GNUBG 2-ply neural-net move evaluation.
8. Show the strongest move and available engine candidates.

The opponent's complete position is therefore an explicit input to every move.

## Deployment

- Python: 3.13
- Main file: `app.py`
- Install dependencies from `requirements.txt`
