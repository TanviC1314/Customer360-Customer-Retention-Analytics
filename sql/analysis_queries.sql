-- =============================================================================
-- Customer360 - SQL analysis (PostgreSQL 16)
--
-- Run after `python -m src.load_to_postgres`. Each query is marked `-- Qnn | title`
-- and ends with a semicolon, so tests/test_sql_analysis.py (and `python -m src.run_sql_analysis`)
-- can execute them one by one.
--
-- Conventions
--   * churn rate = churned customers / customers, shown as a percentage.
--   * "active" = customers with churned = FALSE (still subscribed at the snapshot date).
--   * The dataset is a single snapshot: it has no calendar dates. "Revenue" means the
--     MONTHLY charge on a customer's plan, so monthly totals are MRR-style figures.
--   * Anything labelled "estimate" / "exposure" is a simple, stated-assumption calculation,
--     not a forecast.
-- =============================================================================


-- Q01 | Total customers
-- Business relevance: size of the customer base and the active / churned split that every
-- other figure is measured against.
SELECT COUNT(*)                                  AS total_customers,
       COUNT(*) FILTER (WHERE churned)           AS churned_customers,
       COUNT(*) FILTER (WHERE NOT churned)       AS active_customers
FROM   customers;


-- Q02 | Overall churn rate
-- Business relevance: the headline KPI; benchmark for judging every segment.
SELECT ROUND(100.0 * AVG(churned::int), 2) AS churn_rate_pct,
       ROUND(100.0 * AVG((NOT churned)::int), 2) AS retention_rate_pct
FROM   customers;


-- Q03 | Churn by contract type (JOIN customers + billing)
-- Business relevance: contract length is the strongest single lever for retention.
SELECT b.contract,
       COUNT(*)                                          AS customers,
       SUM(c.churned::int)                               AS churned,
       ROUND(100.0 * AVG(c.churned::int), 2)             AS churn_rate_pct
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
GROUP  BY b.contract
ORDER  BY CASE b.contract WHEN 'Month-to-month' THEN 1 WHEN 'One year' THEN 2 ELSE 3 END;


-- Q04 | Churn by payment method, ranked with a window function
-- Business relevance: shows whether a payment route (e.g. manual vs automatic) signals risk.
SELECT b.payment_method,
       COUNT(*)                                          AS customers,
       SUM(c.churned::int)                               AS churned,
       ROUND(100.0 * AVG(c.churned::int), 2)             AS churn_rate_pct,
       RANK() OVER (ORDER BY AVG(c.churned::int) DESC)   AS risk_rank
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
GROUP  BY b.payment_method
ORDER  BY risk_rank;


-- Q05 | Churn by internet service, with each type's share of all churners
-- Business relevance: locates which product line is losing customers.
SELECT s.internet_service,
       COUNT(*)                                                      AS customers,
       SUM(c.churned::int)                                           AS churned,
       ROUND(100.0 * AVG(c.churned::int), 2)                         AS churn_rate_pct,
       ROUND(100.0 * SUM(c.churned::int) / SUM(SUM(c.churned::int)) OVER (), 1) AS share_of_all_churners_pct
FROM   customers c
JOIN   services  s ON s.customer_id = c.customer_id
GROUP  BY s.internet_service
ORDER  BY churn_rate_pct DESC;


-- Q06 | Average monthly charges by churn status (mean and median)
-- Business relevance: tests whether leavers pay more than stayers (price sensitivity).
SELECT CASE WHEN c.churned THEN 'Churned' ELSE 'Retained' END AS status,
       COUNT(*)                                                AS customers,
       ROUND(AVG(b.monthly_charges), 2)                        AS avg_monthly_charges,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b.monthly_charges))::numeric, 2) AS median_monthly_charges
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
GROUP  BY c.churned
ORDER  BY c.churned;


-- Q07 | Average tenure by churn status
-- Business relevance: churners leave early; tells us when the retention effort must happen.
SELECT CASE WHEN churned THEN 'Churned' ELSE 'Retained' END AS status,
       COUNT(*)                                              AS customers,
       ROUND(AVG(tenure_months), 2)                          AS avg_tenure_months,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tenure_months))::numeric, 1) AS median_tenure_months
FROM   customers
GROUP  BY churned
ORDER  BY churned;


-- Q08 | Customer segmentation: lifecycle stage x spend level (CTE + CASE WHEN)
-- Business relevance: five actionable segments with their size, value and churn so campaigns
-- can be targeted rather than blanket.
WITH segmented AS (
    SELECT v.*,
           CASE
               WHEN tenure_months >= 49                           THEN '1 Loyal (49+ months)'
               WHEN tenure_months >= 13 AND monthly_charges >= 70 THEN '2 Established high-spend'
               WHEN tenure_months >= 13                           THEN '3 Established standard'
               WHEN monthly_charges >= 70                         THEN '4 New high-spend (<=12 months)'
               ELSE                                                    '5 New standard (<=12 months)'
           END AS segment
    FROM   customer_360 v
)
SELECT segment,
       COUNT(*)                                          AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_of_customers_pct,
       ROUND(AVG(monthly_charges), 2)                    AS avg_monthly_charges,
       ROUND(100.0 * AVG(churned::int), 2)               AS churn_rate_pct
FROM   segmented
GROUP  BY segment
ORDER  BY segment;


-- Q09 | High-value customer identification: revenue deciles (window function NTILE)
-- Business relevance: "high value" = top lifetime-revenue decile. Shows how concentrated
-- revenue is and whether valuable customers are also leaving.
WITH ranked AS (
    SELECT customer_id, total_charges, monthly_charges, churned,
           NTILE(10) OVER (ORDER BY total_charges DESC) AS revenue_decile   -- 1 = top 10 %
    FROM   customer_360
)
SELECT revenue_decile,
       COUNT(*)                                                   AS customers,
       ROUND(SUM(total_charges), 0)                               AS lifetime_revenue,
       ROUND(100.0 * SUM(total_charges) / SUM(SUM(total_charges)) OVER (), 1) AS share_of_revenue_pct,
       ROUND(AVG(monthly_charges), 2)                             AS avg_monthly_charges,
       ROUND(100.0 * AVG(churned::int), 2)                        AS churn_rate_pct
FROM   ranked
GROUP  BY revenue_decile
ORDER  BY revenue_decile;


-- Q10 | Tenure-based retention analysis (JOIN to the tenure_bands lookup + cumulative window)
-- Business relevance: where in the lifecycle customers are lost; the cumulative column shows
-- what share of all churn has already happened by the end of each stage.
SELECT t.band_name                                              AS tenure_band,
       COUNT(*)                                                 AS customers,
       SUM(c.churned::int)                                      AS churned,
       ROUND(100.0 * AVG(c.churned::int), 2)                    AS churn_rate_pct,
       ROUND(100.0 * AVG((NOT c.churned)::int), 2)              AS retention_rate_pct,
       ROUND(100.0 * SUM(SUM(c.churned::int)) OVER (ORDER BY t.sort_order)
                   / SUM(SUM(c.churned::int)) OVER (), 1)       AS cumulative_share_of_churners_pct
FROM   customers c
JOIN   tenure_bands t ON c.tenure_months BETWEEN t.min_months AND t.max_months
GROUP  BY t.band_name, t.sort_order
ORDER  BY t.sort_order;


-- Q11 | Service combinations associated with churn (internet customers, n >= 50)
-- Business relevance: finds the bundles that keep customers (and the bare-bones plans that do not).
WITH combos AS (
    SELECT c.churned,
           s.internet_service,
           COALESCE(NULLIF(CONCAT_WS(' + ',
               CASE WHEN s.online_security   THEN 'Security'   END,
               CASE WHEN s.online_backup     THEN 'Backup'     END,
               CASE WHEN s.device_protection THEN 'Protection' END,
               CASE WHEN s.tech_support      THEN 'TechSupport' END,
               CASE WHEN s.streaming_tv OR s.streaming_movies THEN 'Streaming' END), ''), 'No add-ons') AS add_on_bundle
    FROM   customers c
    JOIN   services  s ON s.customer_id = c.customer_id
    WHERE  s.internet_service <> 'No internet'
)
SELECT internet_service,
       add_on_bundle,
       COUNT(*)                              AS customers,
       ROUND(100.0 * AVG(churned::int), 2)   AS churn_rate_pct
FROM   combos
GROUP  BY internet_service, add_on_bundle
HAVING COUNT(*) >= 50
ORDER  BY churn_rate_pct DESC
LIMIT  12;


-- Q12 | Monthly revenue analysis by contract: active MRR, lost MRR and ARPU
-- Business relevance: how much recurring revenue each contract type carries, and how much of
-- it has already walked away. (MRR = sum of monthly plan charges.)
SELECT b.contract,
       ROUND(SUM(b.monthly_charges) FILTER (WHERE NOT c.churned), 0)   AS active_mrr,
       ROUND(SUM(b.monthly_charges) FILTER (WHERE c.churned), 0)       AS lost_mrr,
       ROUND(100.0 * SUM(b.monthly_charges) FILTER (WHERE c.churned) / SUM(b.monthly_charges), 1) AS pct_of_mrr_lost,
       ROUND(AVG(b.monthly_charges), 2)                                AS arpu,
       ROUND(100.0 * SUM(b.monthly_charges) FILTER (WHERE NOT c.churned)
             / SUM(SUM(b.monthly_charges) FILTER (WHERE NOT c.churned)) OVER (), 1) AS share_of_active_mrr_pct
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
GROUP  BY b.contract
ORDER  BY active_mrr DESC;


-- Q13 | Customer groups with higher-than-average churn: relative risk (subquery + HAVING)
-- Business relevance: contract x internet x payment profiles (n >= 100) ranked by how many times
-- more likely they are to churn than the average customer.
SELECT b.contract, s.internet_service, b.payment_method,
       COUNT(*)                                                   AS customers,
       ROUND(100.0 * AVG(c.churned::int), 1)                      AS churn_rate_pct,
       ROUND(AVG(c.churned::int) / (SELECT AVG(churned::int) FROM customers), 2) AS relative_risk
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
JOIN   services  s ON s.customer_id = c.customer_id
GROUP  BY b.contract, s.internet_service, b.payment_method
HAVING COUNT(*) >= 100
   AND AVG(c.churned::int) > (SELECT AVG(churned::int) FROM customers)
ORDER  BY relative_risk DESC
LIMIT  10;


-- Q14 | Senior-citizen churn comparison (conditional aggregation)
-- Business relevance: seniors churn far more; this checks whether their product mix explains it.
SELECT CASE WHEN c.senior_citizen THEN 'Senior' ELSE 'Non-senior' END           AS customer_group,
       COUNT(*)                                                                 AS customers,
       ROUND(100.0 * AVG(c.churned::int), 2)                                    AS churn_rate_pct,
       ROUND(AVG(b.monthly_charges), 2)                                         AS avg_monthly_charges,
       ROUND(100.0 * AVG((s.internet_service = 'Fiber optic')::int), 1)         AS pct_fiber,
       ROUND(100.0 * AVG((b.contract = 'Month-to-month')::int), 1)              AS pct_month_to_month,
       ROUND(100.0 * AVG((b.payment_method = 'Electronic check')::int), 1)      AS pct_electronic_check,
       ROUND(100.0 * AVG((s.tech_support)::int) FILTER (WHERE s.internet_service <> 'No internet'), 1) AS pct_tech_support_among_internet
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
JOIN   services  s ON s.customer_id = c.customer_id
GROUP  BY c.senior_citizen
ORDER  BY c.senior_citizen;


-- Q15 | Paperless billing comparison, controlling for contract (CTE + pivot with FILTER)
-- Business relevance: a naive paperless-vs-paper comparison is confounded by contract type,
-- so churn is compared within each contract.
WITH base AS (
    SELECT b.contract, b.paperless_billing, c.churned
    FROM   customers c
    JOIN   billing   b ON b.customer_id = c.customer_id
)
SELECT contract,
       COUNT(*) FILTER (WHERE paperless_billing)                                          AS paperless_customers,
       ROUND(100.0 * AVG(churned::int) FILTER (WHERE paperless_billing), 2)               AS paperless_churn_pct,
       COUNT(*) FILTER (WHERE NOT paperless_billing)                                      AS paper_customers,
       ROUND(100.0 * AVG(churned::int) FILTER (WHERE NOT paperless_billing), 2)           AS paper_churn_pct,
       ROUND(100.0 * (AVG(churned::int) FILTER (WHERE paperless_billing)
                    - AVG(churned::int) FILTER (WHERE NOT paperless_billing)), 2)         AS gap_pct_points
FROM   base
GROUP  BY contract
UNION ALL
SELECT 'All contracts',
       COUNT(*) FILTER (WHERE paperless_billing),
       ROUND(100.0 * AVG(churned::int) FILTER (WHERE paperless_billing), 2),
       COUNT(*) FILTER (WHERE NOT paperless_billing),
       ROUND(100.0 * AVG(churned::int) FILTER (WHERE NOT paperless_billing), 2),
       ROUND(100.0 * (AVG(churned::int) FILTER (WHERE paperless_billing)
                    - AVG(churned::int) FILTER (WHERE NOT paperless_billing)), 2)
FROM   base;


-- Q16 | Revenue at risk: realised loss and current exposure (CTEs + subquery)
-- Business relevance: sizes the problem in money. Two figures:
--   realised_lost_mrr : monthly charges of customers who already churned (a fact).
--   exposed_mrr       : monthly charges of ACTIVE customers whose profile (contract x internet x
--                       payment) has a historical churn rate >= 40 % - an exposure measure,
--                       NOT a forecast of what will actually be lost. Annualised = x12.
WITH profile_rates AS (
    SELECT b.contract, s.internet_service, b.payment_method,
           COUNT(*) AS n,
           AVG(c.churned::int) AS profile_churn_rate
    FROM   customers c
    JOIN   billing   b ON b.customer_id = c.customer_id
    JOIN   services  s ON s.customer_id = c.customer_id
    GROUP  BY b.contract, s.internet_service, b.payment_method
),
active AS (
    SELECT b.contract, s.internet_service, b.payment_method, b.monthly_charges
    FROM   customers c
    JOIN   billing   b ON b.customer_id = c.customer_id
    JOIN   services  s ON s.customer_id = c.customer_id
    WHERE  NOT c.churned
)
SELECT (SELECT ROUND(SUM(b.monthly_charges), 0) FROM billing b JOIN customers c USING (customer_id) WHERE c.churned)  AS realised_lost_mrr,
       (SELECT ROUND(SUM(b.monthly_charges), 0) FROM billing b JOIN customers c USING (customer_id) WHERE NOT c.churned) AS active_mrr,
       COUNT(*)                                       AS exposed_active_customers,
       ROUND(SUM(a.monthly_charges), 0)               AS exposed_mrr,
       ROUND(SUM(a.monthly_charges) * 12, 0)          AS exposed_annualised_revenue,
       ROUND(100.0 * SUM(a.monthly_charges)
             / (SELECT SUM(b.monthly_charges) FROM billing b JOIN customers c USING (customer_id) WHERE NOT c.churned), 1) AS exposed_pct_of_active_mrr
FROM   active a
JOIN   profile_rates p USING (contract, internet_service, payment_method)
WHERE  p.n >= 30 AND p.profile_churn_rate >= 0.40;


-- Q17 | Customer lifetime value (CLV) estimate by contract type (ESTIMATE)
-- Business relevance: puts a monetary value on keeping a customer, by contract.
-- Method: monthly churn hazard = churn events / total customer-months observed (exponential
-- survival MLE). Expected lifetime over a 60-month horizon = (1 - e^(-h*60)) / h.
-- CLV_60m = ARPU x expected lifetime. Undiscounted; assumes constant hazard and constant ARPU.
WITH hazard AS (
    SELECT b.contract,
           COUNT(*)                                     AS customers,
           SUM(c.churned::int)                          AS churn_events,
           SUM(c.tenure_months)                         AS customer_months,
           AVG(b.monthly_charges)                       AS arpu,
           AVG(b.total_charges)                         AS avg_revenue_to_date,
           SUM(c.churned::int)::numeric / NULLIF(SUM(c.tenure_months), 0) AS monthly_hazard
    FROM   customers c
    JOIN   billing   b ON b.customer_id = c.customer_id
    GROUP  BY b.contract
)
SELECT contract,
       customers,
       ROUND(arpu, 2)                                                   AS arpu,
       ROUND(100 * monthly_hazard, 2)                                   AS monthly_churn_hazard_pct,
       ROUND((1 - EXP(-monthly_hazard * 60)) / monthly_hazard, 1)       AS expected_lifetime_months_60m_cap,
       ROUND(arpu * (1 - EXP(-monthly_hazard * 60)) / monthly_hazard, 0) AS estimated_clv_60m,
       ROUND(avg_revenue_to_date, 0)                                    AS avg_revenue_to_date
FROM   hazard
ORDER  BY estimated_clv_60m;


-- Q18 | Customer retention opportunities: active customers without support services (CTEs)
-- Business relevance: customers who hold no protection/support service churn far more than those
-- who do, within the same contract. This sizes the active base that a "add a support service"
-- offer could target. The gap is an OBSERVED difference, not proof that adding the service
-- causes lower churn, so "illustrative_mrr_protected" is an upper-bound illustration.
WITH rates AS (
    SELECT b.contract,
           AVG(c.churned::int) FILTER (WHERE v.support_services = 0) AS churn_without_support,
           AVG(c.churned::int) FILTER (WHERE v.support_services > 0) AS churn_with_support
    FROM   customers c
    JOIN   billing b ON b.customer_id = c.customer_id
    JOIN   customer_360 v ON v.customer_id = c.customer_id
    WHERE  v.internet_service <> 'No internet'
    GROUP  BY b.contract
),
targets AS (
    SELECT v.contract, COUNT(*) AS active_customers_without_support, SUM(v.monthly_charges) AS mrr
    FROM   customer_360 v
    WHERE  NOT v.churned AND v.internet_service <> 'No internet' AND v.support_services = 0
    GROUP  BY v.contract
)
SELECT t.contract,
       t.active_customers_without_support,
       ROUND(t.mrr, 0)                                              AS mrr_of_target_customers,
       ROUND(100 * r.churn_without_support, 1)                      AS churn_pct_without_support,
       ROUND(100 * r.churn_with_support, 1)                         AS churn_pct_with_support,
       ROUND(100 * (r.churn_without_support - r.churn_with_support), 1) AS gap_pct_points,
       ROUND(t.mrr * (r.churn_without_support - r.churn_with_support), 0) AS illustrative_mrr_protected
FROM   targets t
JOIN   rates r USING (contract)
ORDER  BY illustrative_mrr_protected DESC;


-- Q19 | Churn by multiple customer attributes with subtotals (GROUP BY ROLLUP)
-- Business relevance: one result set gives contract x internet detail AND the contract / grand
-- totals, so the pattern (fiber + month-to-month is the worst cell) is visible against its context.
SELECT COALESCE(b.contract, 'ALL CONTRACTS')               AS contract,
       COALESCE(s.internet_service, 'all internet types')  AS internet_service,
       COUNT(*)                                            AS customers,
       SUM(c.churned::int)                                 AS churned,
       ROUND(100.0 * AVG(c.churned::int), 1)               AS churn_rate_pct,
       GROUPING(b.contract, s.internet_service)            AS grouping_level   -- 0 detail, 1 contract subtotal, 3 grand total
FROM   customers c
JOIN   billing   b ON b.customer_id = c.customer_id
JOIN   services  s ON s.customer_id = c.customer_id
GROUP  BY ROLLUP (b.contract, s.internet_service)
ORDER  BY GROUPING(b.contract), b.contract, GROUPING(s.internet_service), churn_rate_pct DESC;


-- Q20 | Ranking customer segments by churn rate + Pareto share of churn (window functions)
-- Business relevance: shows how few segments account for most churners, i.e. where to focus first.
WITH seg AS (
    SELECT b.contract, v.tenure_band, v.internet_service,
           COUNT(*)              AS customers,
           SUM(v.churned::int)   AS churned
    FROM   customer_360 v
    JOIN   billing b ON b.customer_id = v.customer_id
    GROUP  BY b.contract, v.tenure_band, v.internet_service
    HAVING COUNT(*) >= 50
)
SELECT RANK() OVER (ORDER BY churned::numeric / customers DESC)          AS churn_rank,
       contract, tenure_band, internet_service, customers, churned,
       ROUND(100.0 * churned / customers, 1)                             AS churn_rate_pct,
       ROUND(100.0 * SUM(churned) OVER (ORDER BY churned::numeric / customers DESC, customers DESC)
                   / SUM(churned) OVER (), 1)                            AS cumulative_share_of_segment_churners_pct
FROM   seg
ORDER  BY churn_rank
LIMIT  12;


-- Q21 | Retention watch-list: top active at-risk customers per contract (ROW_NUMBER + PARTITION BY)
-- Business relevance: an operational list for the retention team - the highest-paying ACTIVE
-- customers who match the riskiest profile (fiber, electronic check, <= 24 months, no support).
WITH risky AS (
    SELECT customer_id, contract, tenure_months, monthly_charges, payment_method, internet_service,
           ROW_NUMBER() OVER (PARTITION BY contract ORDER BY monthly_charges DESC, customer_id) AS rn
    FROM   customer_360
    WHERE  NOT churned
      AND  internet_service = 'Fiber optic'
      AND  payment_method   = 'Electronic check'
      AND  tenure_months   <= 24
      AND  support_services = 0
)
SELECT contract, rn AS rank_in_contract, customer_id, tenure_months, monthly_charges
FROM   risky
WHERE  rn <= 3
ORDER  BY contract, rn;


-- Q22 | Add-on depth vs churn: number of support services held (JOIN + CASE + running comparison)
-- Business relevance: each extra protection/support service is associated with lower churn;
-- supports bundling as a retention play. Restricted to internet customers.
SELECT v.support_services                                       AS support_services_held,
       COUNT(*)                                                 AS customers,
       ROUND(100.0 * AVG(v.churned::int), 2)                    AS churn_rate_pct,
       ROUND(AVG(v.monthly_charges), 2)                         AS avg_monthly_charges,
       ROUND(100.0 * AVG(v.churned::int)
             - LAG(100.0 * AVG(v.churned::int)) OVER (ORDER BY v.support_services), 2) AS change_vs_previous_level_pts
FROM   customer_360 v
WHERE  v.internet_service <> 'No internet'
GROUP  BY v.support_services
ORDER  BY v.support_services;
