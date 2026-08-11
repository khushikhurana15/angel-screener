# tests/

Manual verification scripts used during development to test individual
components in isolation before wiring them into the live pipeline:

- `test_connection.py` — Angel One SmartAPI login/session check
- `test_smma.py` — SMMA(20)/SMMA(120) calculation + crossover detection
- `test_etq_depth.py` — ETQ (5/20/60 min) and market depth fetch
- `test_groq_explain.py` — Groq AI explanation generation (with fallback)
- `test_batch_price.py` — batch LTP/quote fetching

Not an automated pytest suite — these are one-off diagnostic scripts,
run manually from the project root, e.g.:

    python tests/test_smma.py

