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
    # ---- Cup competitions (added 2026-07-13) — same pipeline; their matches carry
    # a `stage` (ESPN season.slug) that feeds the is_knockout meta covariate, and
    # club form pulls cross-competition rows automatically (shared team_ids).
    # UEFA cups: ESPN splits qualifying into its own feed code — both codes merge
    # into the one Sandy league (ingest dedupes by event_id).
    "ucl": (("uefa.champions", "uefa.champions_qual"), "Champions League", "⭐", set(range(1, 13))),  # quals Jul–Aug, league Sep–Jan, KO Feb–Jun
    "uel": (("uefa.europa", "uefa.europa_qual"), "Europa League", "🇪🇺", set(range(1, 13))),          # quals Jul–Aug, league Sep–Jan, KO Feb–May
    "ccc": ("concacaf.champions", "Concacaf Champions Cup", "🌎", {2, 3, 4, 5, 6}),  # Feb–Jun
    "lgc": ("concacaf.leagues.cup", "Leagues Cup", "🇺🇸🇲🇽", {7, 8, 9}),              # Jul–Sep (2026: Aug 4–Sep 6)
    "lib": ("conmebol.libertadores", "Copa Libertadores", "🏆", set(range(2, 12))),  # Feb–Nov
    "sud": ("conmebol.sudamericana", "Copa Sudamericana", "🥈", set(range(2, 12))),  # Feb–Nov
}
