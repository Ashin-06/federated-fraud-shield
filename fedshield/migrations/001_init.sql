-- Published model versions. Weights and normalisation statistics are stored as base64 of
-- little-endian float32 arrays, not pickles, so loading a model never runs code.
CREATE TABLE models (
    version BIGINT PRIMARY KEY,
    created_at TEXT NOT NULL,
    trained_by TEXT NOT NULL,
    config TEXT NOT NULL,
    metrics TEXT NOT NULL,
    epsilon TEXT NOT NULL,
    d_in BIGINT NOT NULL,
    weights TEXT NOT NULL,
    mean TEXT NOT NULL,
    std TEXT NOT NULL,
    threshold TEXT NOT NULL
);

-- One row per scoring call: who asked, which model version, how many rows, how many flagged.
CREATE TABLE score_log (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    version BIGINT NOT NULL,
    rows_scored BIGINT NOT NULL,
    flagged BIGINT NOT NULL
);
