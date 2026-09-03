# Data

`all_raw.csv` — match results and bookmaker odds for the English Premier League,
combined from the per-season CSV files published by **Football-Data.co.uk**
(https://www.football-data.co.uk/englandm.php).

- One row per match. The `source_file` column records the season file each row came from.
- The analyses in this repository use `Div == "E0"` (Premier League) between
  **1 August 2019** and **31 December 2025**, which gives **2,430 matches**.
- Odds columns follow the Football-Data convention. Closing odds carry a `C`:
  `B365H/D/A` are Bet365 opening prices, `B365CH/CD/CA` are Bet365 closing prices,
  `PSCH/PSCD/PSCA` Pinnacle closing, `WHCH/WHCD/WHCA` William Hill closing,
  `BWCH/BWCD/BWCA` Betway closing.
- No values are imputed. A match is used in an analysis only if that analysis's
  required odds are present and numeric.

The data are redistributed here purely to make the paper's results reproducible.
Football-Data.co.uk makes this data freely available for academic use; please
credit them as the original source.
