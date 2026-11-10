#!/usr/bin/env bash
# Download the bwin "Population Trends" sports-betting dataset from the Transparency
# Project (Division on Addiction, Cambridge Health Alliance, a Harvard Medical School
# teaching hospital). The data is public; it is not redistributed in this repo.
set -euo pipefail

DEST="${1:-data/raw}"
BASE="http://www.thetransparencyproject.org"
mkdir -p "$DEST"

for f in PopTrendsBData1Fixed.csv PopTrendsBData2Live.csv PopTrendsBData3Aggs.csv; do
  if [ -s "$DEST/$f" ]; then
    echo "have $f"
  else
    echo "downloading $f"
    curl -fsSL --retry 3 -o "$DEST/$f" "$BASE/datasets/$f"
  fi
done
curl -fsSL --retry 3 -o "$DEST/codebook.pdf" "$BASE/codebooks/PopTrendsBCodebook20210303.pdf"

wc -l "$DEST"/PopTrendsB*.csv
