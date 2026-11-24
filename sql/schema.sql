-- =============================================================================
-- Customer360 - PostgreSQL schema
-- Re-runnable: drops and recreates all objects. Loaded by `python -m src.load_to_postgres`.
--
-- Model (3NF-style split of the 21-column Telco file into three 1:1 tables so that
-- queries can demonstrate real JOINs, plus a small lookup table for lifecycle bands):
--
--   customers (PK customer_id)      who the customer is + tenure + churn outcome
--   services  (PK/FK customer_id)   what they subscribe to
--   billing   (PK/FK customer_id)   how they are billed and what they pay
--   tenure_bands                    lookup: lifecycle stage by months of tenure
--   customer_360 (view)             one flat row per customer joining all of the above
-- =============================================================================

DROP VIEW  IF EXISTS customer_360;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS tenure_bands;

CREATE TABLE tenure_bands (
    band_name   VARCHAR(20) PRIMARY KEY,
    min_months  SMALLINT NOT NULL,
    max_months  SMALLINT NOT NULL,
    sort_order  SMALLINT NOT NULL UNIQUE,
    CHECK (min_months <= max_months)
);

INSERT INTO tenure_bands (band_name, min_months, max_months, sort_order) VALUES
    ('0-12 months',   0, 12, 1),
    ('13-24 months', 13, 24, 2),
    ('25-48 months', 25, 48, 3),
    ('49-72 months', 49, 72, 4);

CREATE TABLE customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    gender          VARCHAR(6)  NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen  BOOLEAN     NOT NULL,
    partner         BOOLEAN     NOT NULL,
    dependents      BOOLEAN     NOT NULL,
    tenure_months   SMALLINT    NOT NULL CHECK (tenure_months BETWEEN 0 AND 72),
    churned         BOOLEAN     NOT NULL
);

CREATE TABLE services (
    customer_id        VARCHAR(20) PRIMARY KEY REFERENCES customers (customer_id) ON DELETE CASCADE,
    phone_service      BOOLEAN     NOT NULL,
    multiple_lines     BOOLEAN     NOT NULL,
    internet_service   VARCHAR(12) NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No internet')),
    online_security    BOOLEAN     NOT NULL,
    online_backup      BOOLEAN     NOT NULL,
    device_protection  BOOLEAN     NOT NULL,
    tech_support       BOOLEAN     NOT NULL,
    streaming_tv       BOOLEAN     NOT NULL,
    streaming_movies   BOOLEAN     NOT NULL,
    -- add-ons only exist for internet subscribers
    CHECK (internet_service <> 'No internet'
           OR NOT (online_security OR online_backup OR device_protection
                   OR tech_support OR streaming_tv OR streaming_movies)),
    -- multiple lines requires a phone line
    CHECK (phone_service OR NOT multiple_lines)
);

CREATE TABLE billing (
    customer_id        VARCHAR(20) PRIMARY KEY REFERENCES customers (customer_id) ON DELETE CASCADE,
    contract           VARCHAR(14) NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    paperless_billing  BOOLEAN     NOT NULL,
    payment_method     VARCHAR(30) NOT NULL CHECK (payment_method IN
                         ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')),
    monthly_charges    NUMERIC(7, 2) NOT NULL CHECK (monthly_charges > 0),
    total_charges      NUMERIC(9, 2) NOT NULL CHECK (total_charges >= 0)
);

-- Foreign-key columns are the primary keys themselves, so joins are index-backed.
-- Helper indexes for the attributes used most in GROUP BY / filters:
CREATE INDEX idx_customers_churned  ON customers (churned);
CREATE INDEX idx_billing_contract   ON billing (contract);
CREATE INDEX idx_services_internet  ON services (internet_service);

CREATE VIEW customer_360 AS
SELECT  c.customer_id,
        c.gender,
        c.senior_citizen,
        c.partner,
        c.dependents,
        c.tenure_months,
        t.band_name AS tenure_band,
        t.sort_order AS tenure_band_order,
        c.churned,
        s.phone_service,
        s.multiple_lines,
        s.internet_service,
        s.online_security,
        s.online_backup,
        s.device_protection,
        s.tech_support,
        s.streaming_tv,
        s.streaming_movies,
        (s.online_security::int + s.online_backup::int + s.device_protection::int + s.tech_support::int) AS support_services,
        b.contract,
        b.paperless_billing,
        b.payment_method,
        b.monthly_charges,
        b.total_charges
FROM    customers c
JOIN    services s      ON s.customer_id = c.customer_id
JOIN    billing  b      ON b.customer_id = c.customer_id
JOIN    tenure_bands t  ON c.tenure_months BETWEEN t.min_months AND t.max_months;

COMMENT ON VIEW customer_360 IS 'Flat customer view joining customers, services, billing and tenure bands.';
COMMENT ON COLUMN customers.churned IS 'TRUE if the customer left within the last month (source column Churn).';
COMMENT ON COLUMN billing.total_charges IS 'Cumulative charges to date; 0 for customers with 0 months tenure (source value was blank).';
