-- docs/architecture.md does not exist in this repository (checked at generation
-- time), so its §5 could not be read. This schema is inferred instead from
-- CLAUDE.md (data schemas, compliance/explain/ranking feature descriptions)
-- and standard patterns for the five named tables. Reconcile against the real
-- architecture doc once it exists.

CREATE TABLE IF NOT EXISTS products (
    article_id          BIGINT PRIMARY KEY,
    prod_name           TEXT NOT NULL,
    product_type_name   TEXT NOT NULL,
    product_group_name  TEXT NOT NULL,
    category_l1         TEXT NOT NULL,
    colour_group_name   TEXT,
    index_group_name    TEXT,
    department_name     TEXT,
    detail_desc         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_products_category_l1 ON products (category_l1);
CREATE INDEX IF NOT EXISTS idx_products_product_group_name ON products (product_group_name);

CREATE TABLE IF NOT EXISTS complements (
    article_id             BIGINT NOT NULL REFERENCES products (article_id) ON DELETE CASCADE,
    complement_article_id  BIGINT NOT NULL REFERENCES products (article_id) ON DELETE CASCADE,
    score                   DOUBLE PRECISION NOT NULL,
    method                  TEXT NOT NULL,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, complement_article_id)
);

CREATE INDEX IF NOT EXISTS idx_complements_complement_article_id
    ON complements (complement_article_id);

CREATE TABLE IF NOT EXISTS users (
    customer_id         TEXT PRIMARY KEY,
    age                 REAL,
    postal_code         TEXT,
    club_member_status  TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_postal_code ON users (postal_code);

CREATE TABLE IF NOT EXISTS consent (
    consent_id     BIGSERIAL PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES users (customer_id) ON DELETE CASCADE,
    consent_type   TEXT NOT NULL,
    granted        BOOLEAN NOT NULL,
    granted_at     TIMESTAMPTZ,
    revoked_at     TIMESTAMPTZ,
    source         TEXT
);

CREATE INDEX IF NOT EXISTS idx_consent_customer_id ON consent (customer_id);

-- Only one active (non-revoked) grant per customer/consent_type at a time.
CREATE UNIQUE INDEX IF NOT EXISTS uq_consent_customer_type_current
    ON consent (customer_id, consent_type)
    WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS recommendation_log (
    log_id          BIGSERIAL PRIMARY KEY,
    request_id      UUID NOT NULL,
    customer_id     TEXT REFERENCES users (customer_id) ON DELETE SET NULL,
    session_id      TEXT,
    article_id      BIGINT NOT NULL REFERENCES products (article_id) ON DELETE CASCADE,
    rank_position   INTEGER NOT NULL,
    score           DOUBLE PRECISION,
    reason_code     TEXT,
    model_version   TEXT NOT NULL,
    served_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recommendation_log_customer_id ON recommendation_log (customer_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_log_request_id ON recommendation_log (request_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_log_served_at ON recommendation_log (served_at);
