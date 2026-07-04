"""Sandy multi-league soccer vertical — Colombia, México, España, Inglaterra.
Same architecture as the MLS vertical (ESPN data, Dixon-Coles goals + corners,
double-chance / totals / corners markets, walk-forward calibration, meta-model),
with `league` as a first-class column so all four share one schema and pipeline."""

# league key → (espn code, display name, flag emoji, months with league play)
LEAGUES = {
    "col": ("col.1", "Liga Colombia", "🇨🇴", set(range(1, 13))),      # Feb–Dec (apertura/clausura)
    "mex": ("mex.1", "Liga MX", "🇲🇽", set(range(1, 13))),            # Jul–May (apertura/clausura)
    "esp": ("esp.1", "La Liga", "🇪🇸", {1, 2, 3, 4, 5, 8, 9, 10, 11, 12}),
    "eng": ("eng.1", "Premier League", "🏴", {1, 2, 3, 4, 5, 8, 9, 10, 11, 12}),
}
