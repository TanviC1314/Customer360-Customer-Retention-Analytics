-- =============================================================================
-- Customer360 - data validation queries
-- Each query returns (check_name, violations). A healthy load has violations = 0
-- for every row (except the informational row counts). Executed automatically by
-- `python -m src.load_to_postgres` and by tests/test_sql_analysis.py.
-- =============================================================================

SELECT 'customers row count must be 7043' AS check_name,
       ABS((SELECT COUNT(*) FROM customers) - 7043) AS violations
UNION ALL
SELECT 'services and billing have one row per customer',
       (SELECT COUNT(*) FROM customers c
          LEFT JOIN services s ON s.customer_id = c.customer_id
          LEFT JOIN billing  b ON b.customer_id = c.customer_id
         WHERE s.customer_id IS NULL OR b.customer_id IS NULL)
UNION ALL
SELECT 'no orphan rows in services / billing',
       (SELECT COUNT(*) FROM services s WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = s.customer_id))
     + (SELECT COUNT(*) FROM billing  b WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = b.customer_id))
UNION ALL
SELECT 'customer_id is unique',
       (SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM customers)
UNION ALL
SELECT 'every customer maps to exactly one tenure band (view row count = 7043)',
       ABS((SELECT COUNT(*) FROM customer_360) - 7043)
UNION ALL
SELECT 'churned customers must be 1869 (matches source file)',
       ABS((SELECT COUNT(*) FROM customers WHERE churned) - 1869)
UNION ALL
SELECT 'no NULLs in key analytical columns',
       (SELECT COUNT(*) FROM customer_360
         WHERE tenure_months IS NULL OR monthly_charges IS NULL OR total_charges IS NULL
            OR contract IS NULL OR churned IS NULL OR internet_service IS NULL)
UNION ALL
SELECT 'total_charges is zero only for zero-tenure customers',
       (SELECT COUNT(*) FROM billing b JOIN customers c USING (customer_id)
         WHERE (b.total_charges = 0) <> (c.tenure_months = 0))
UNION ALL
SELECT 'zero-tenure customers have not churned (source data property)',
       (SELECT COUNT(*) FROM customers WHERE tenure_months = 0 AND churned)
UNION ALL
SELECT 'no internet add-ons without internet service',
       (SELECT COUNT(*) FROM services
         WHERE internet_service = 'No internet'
           AND (online_security OR online_backup OR device_protection OR tech_support OR streaming_tv OR streaming_movies))
UNION ALL
SELECT 'monthly charges within plausible range (18-120)',
       (SELECT COUNT(*) FROM billing WHERE monthly_charges NOT BETWEEN 18 AND 120)
UNION ALL
SELECT 'total charges never exceed monthly x tenure x 1.6 (sanity bound)',
       (SELECT COUNT(*) FROM billing b JOIN customers c USING (customer_id)
         WHERE b.total_charges > b.monthly_charges * c.tenure_months * 1.6);
