# Bundled webfonts

Self-hosted so the GUI has no runtime CDN dependency.

| File | Family | Source | Licence |
| --- | --- | --- | --- |
| `instrument-sans-latin.woff2`, `instrument-sans-latin-ext.woff2` | Instrument Sans (variable, 400–700) | Google Fonts (`fonts.gstatic.com`) | SIL Open Font License 1.1 |
| `jetbrains-mono-latin.woff2` | JetBrains Mono (variable, 400–600) | Google Fonts (`fonts.gstatic.com`) | SIL Open Font License 1.1 |

Only the latin / latin-ext subsets are bundled. The `unicode-range` values in
`public/index.html` match the subsets Google Fonts ships, so the browser skips a
file it does not need.
