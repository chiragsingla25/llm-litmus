"""SQLite DDL for the runs/responses/scores tables."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id TEXT NOT NULL,
  target_model TEXT NOT NULL,
  target_quantization TEXT,
  judge_model TEXT,
  judge_quantization TEXT,
  prompt_template_version TEXT NOT NULL,
  temperature REAL NOT NULL DEFAULT 0.0,
  sample_size INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS responses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  item_index INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  response_text TEXT NOT NULL,
  gold_reference TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (run_id, item_index)
);
CREATE INDEX IF NOT EXISTS idx_responses_run_id ON responses(run_id);

CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  response_id INTEGER NOT NULL REFERENCES responses(id),
  scorer_version TEXT NOT NULL,
  score REAL,
  passed INTEGER,
  rationale TEXT,
  raw_judge_output TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_scores_response_id ON scores(response_id);
"""
