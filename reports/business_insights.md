# Customer360 - Business insights report

_Customer retention and churn analysis of the IBM Telco Customer Churn sample dataset (7,043 customers). Every figure below was computed
by the code in this repository (Python, PostgreSQL) and is reproducible; see the sources in brackets._

## 1. Executive summary

* **26.5% of customers have churned** (1,869 of 7,043). The monthly recurring revenue (MRR) already lost is **$139K per month**,
  against **$317K** still being billed from active customers [SQL Q01, Q02, Q12].
* Churn is **highly concentrated**. Month-to-month customers are 55% of the base but **89% of churners** (42.7% churn vs 11.3% on
  one-year and 2.8% on two-year contracts). Fibre-optic customers churn at 41.9% (DSL 19.0%) and supply **69%** of churners.
  The single worst group - **new fibre customers on month-to-month contracts** - churns at **70.2%** and alone accounts for about **a third of all churners**.
* Churn happens **early**: 55% of churners left within 12 months of joining; median tenure of churners is 10 months vs 38 for retained customers.
* **Protection and support services** (tech support, online security) are associated with churn of ~15% vs ~42% without them. Electronic-check payers
  churn at 45% vs 15-19% for other payment methods.
* A churn model (XGBoost, ROC-AUC 0.84 on a hold-out test set) ranks customers well enough to be useful: the **top 30% of customers by predicted risk contain 66% of
  the churners that left** (random targeting would reach 30%). Logistic regression performs almost identically, so a simple, explainable model is sufficient.
* The five retention levers in section 7 are sized on active customers. They overlap and are *associations, not causal effects*, so they should be tested with
  controlled pilots before being budgeted.

## 2. Business problem

Subscription businesses lose revenue every time a customer leaves, and winning a new customer usually costs more than keeping one. This project asks:
why do customers leave, which segments leave most, which products and contracts are involved, how do tenure and price matter, which customers are most at
risk now, and what should a retention team do about it.

## 3. Dataset overview

* **Source:** IBM Telco Customer Churn sample dataset (Kaggle `blastchar/telco-customer-churn`; file obtained from IBM's public GitHub mirror, see `data/raw/README.md`).
  Fictional telecom customers of a California operator: 7,043 customers x 21 columns.
* **Content:** demographics (gender, senior citizen, partner, dependents), account (tenure, contract, paperless billing, payment method),
  services (phone, multiple lines, internet type, six add-ons), charges (monthly, total) and the target `Churn` (customers who left in the last month).
* **Limits:** one snapshot, no dates, no cost data, no reasons for leaving; a sample dataset rather than a real operator's data.

## 4. Data preparation (details: `reports/data_cleaning_report.md`)

| Issue | Decision |
|---|---|
| `TotalCharges` read as text; 11 blank values | Converted to numeric. All 11 blanks are customers with tenure 0 who have not been billed yet and none churned, so set to **0.0**. Rows kept. |
| No duplicate rows or customer IDs | Uniqueness enforced by validation and the database primary key |
| `"No internet service"` / `"No phone service"` repeated across 7 columns | Mapped to `"No"`; the information remains in `internet_service` / `phone_service` |
| `SeniorCitizen` 0/1 vs Yes/No elsewhere | Converted to Yes/No |
| Range and consistency checks (tenure 0-72, charges > 0, add-ons only with internet, etc.) | All passed; nothing removed or capped |

Result: 7,043 rows, no missing values, reconciled to the raw file (row count, churn count, and charge totals identical).
The raw file is checksum-verified by the test suite and never modified.

## 5. Key churn findings

For each finding: **what the data shows -> why it matters -> action**. (Charts: `reports/figures/`, notebook `02_customer_churn_analysis.ipynb`.)

1. **Contract type.** Churn 42.7% / 11.3% / 2.8% (month-to-month / one-year / two-year). *Matters:* contract is the strongest driver (Cramer's V 0.41; top model feature).
   *Action:* incentivise migration to 12/24-month contracts early in the lifecycle.
2. **Tenure.** Churn 47.4% in the first year, 28.7%, 20.4%, 9.5% after four years. *Matters:* retention effort pays off most in year one.
   *Action:* onboarding journey, 30/90-day check-ins, early contract offers.
3. **Internet service.** Fibre 41.9%, DSL 19.0%, none 7.4%; the gap persists within each contract type. *Matters:* the premium product is the one losing customers.
   *Action:* review fibre service quality and price-for-value; use customer feedback to find the cause (the data cannot).
4. **Payment method.** Electronic check 45.3% vs 15.2-19.1% for others; still 53.7% vs 32-34% among month-to-month customers only. *Matters:* manual payments give a regular decision point.
   *Action:* incentivise automatic payment.
5. **Price level.** Churners pay $74.44 vs $61.27 per month on average; churn 10.9% under $35 vs 38.0% at $70-90. Combined with tenure: **new customers paying $90+ churn at 75.8%**,
   long-standing customers paying under $35 at 1.3%. *Matters:* price sensitivity is concentrated in new customers. *Action:* target value-adds/price reassurance at new high-spend customers rather than blanket discounts.
6. **Seniors.** 41.7% vs 23.6% (more fibre, more month-to-month, less tech support: 23.9% vs 40.3%). *Action:* support-focused onboarding for senior customers.
7. **Support services.** Internet customers with tech support churn at 15.2% vs 41.6%; churn falls from 56.7% (no support service) to 5.3% (all four). Streaming add-ons show only a small difference (30-34%).
   *Action:* bundle/trial protection and support services; verify with an A/B test.
8. **Not drivers:** gender is essentially unrelated to churn (Cramer's V < 0.01); do not segment on it.

Customer value context [SQL Q09, Q17]: the top 10% of customers by lifetime revenue generate 30.8% of all revenue and churn at 12.5% - lower than average,
but each loss is costly. A simple estimate of 5-year customer lifetime value (undiscounted, constant churn hazard) is about **$2.1K for a month-to-month customer
vs $3.6K for one- or two-year customers**.

## 6. Segments requiring attention

| Segment | Customers | Churn rate | Note |
|---|---|---|---|
| Month-to-month, fibre, tenure 0-12 months | 916 | **70.2%** | 643 churners = 34% of all churn (SQL Q20) |
| New high-spend (<=12 months, >=$70/month) | 880 | **68.1%** | highest-risk lifecycle x spend segment (SQL Q08) |
| Month-to-month + fibre + electronic check | 1,307 | **60.4%** | 2.3x average risk (SQL Q13) |
| Month-to-month, fibre, 13-24 months | 425 | 50.6% | risk persists into year two |
| Internet customers with no support service | 1,267 | 56.7% | vs 24.4% for those with 1+ support service (SQL Q22) |
| Senior citizens | 1,142 | 41.7% | 70.7% on month-to-month contracts |

**Revenue at risk.** Two complementary measures (neither is a forecast):
* *Profile-based exposure* [SQL Q16]: 1,248 active customers belong to contract x internet x payment profiles with a historical churn rate >= 40%; they bill **$99K per month
  (31% of active MRR, about $1.19M annualised)**.
* *Model-based exposure* [dashboard data]: 519 active customers are scored "High" risk (out-of-fold probability >= 0.5), billing **$39.6K per month**.
  The model measure is much smaller because it flags customers individually (a segment's average risk is not every member's risk).

## 7. Machine-learning results (details: `reports/model_results.md`)

* Three model families (logistic regression baseline, random forest, XGBoost), each with and without class weighting, trained with leak-free pipelines,
  a stratified 80/20 split and 5-fold CV. Held-out test set: 1,409 customers.
* All models reach **ROC-AUC ~0.84** (test 0.842-0.844) - the differences between models are smaller than fold-to-fold noise. XGBoost was selected by the
  predefined rule (highest CV ROC-AUC) but logistic regression is statistically indistinguishable.
* **Accuracy alone is misleading:** a model predicting "nobody churns" scores 73.5% accuracy and catches 0% of churners. Unweighted XGBoost: 80.3% accuracy but recall 52.1%.
  Class-weighted XGBoost: accuracy 74.7%, **recall 79.1%**, precision 51.6%.
* With the decision threshold chosen on training data only (0.32), XGBoost gives precision 53.5%, recall 74.3%, F1 0.62 on the test set.
* **Targeting value:** the top 10% of customers by predicted risk contain 29% of churners (lift 2.9x); the top 30% contain 66%.
* **Top drivers** (permutation importance): contract, tenure, internet service, then total charges, paperless billing, payment method and online security -
  matching the EDA.

## 8. Retention recommendations

Sized on *active* customers (overlapping groups; the gap is the observed difference in churn rate versus a comparison group, not a causal effect, so "illustrative MRR protected" is an upper-bound illustration):

| # | Lever | Active customers | Active MRR | Observed churn vs comparison | Suggested action |
|---|---|---|---|---|---|
| 1 | Move month-to-month customers to 12/24-month contracts | 2,220 | $136K | 42.7% vs 6.8% | Discount or perk for commitment, offered in months 3-9; start with high-spend and fibre customers |
| 2 | Review fibre price-for-value and service quality | 1,799 | $169K | 41.9% vs 19.0% (DSL) | Investigate outages/complaints/competitor pricing; pilot a fibre retention offer |
| 3 | Move electronic-check payers to automatic payment | 1,294 | $96K | 45.3% vs 16.0% (auto-pay) | One-off bill credit for switching |
| 4 | Strengthen first-year onboarding | 1,149 | $54K | 47.4% vs 17.1% (13+ months) | Welcome journey, 30/90-day check-ins, proactive follow-up |
| 5 | Add a protection/support service for internet customers lacking one | 549 | $35K | 56.7% vs 24.4% | Free 3-month trial of security/tech support, then reduced price |

**Operational use of the model:** score customers monthly, contact the highest-risk decile first, and **measure** every campaign against a holdout group. Choose the score threshold by
the real cost of a retention offer versus the value of a retained customer - those costs are not in this dataset.

**Pilot first:** the new month-to-month fibre group (916 customers) is large enough to measure results and accounts for a third of churn.

## 9. Limitations

* **Snapshot only.** No dates: no trend, seasonality, survival analysis or true time-based validation. "Revenue" means monthly plan charges (MRR), not booked revenue over time.
* **Association, not causation.** Support services, contract type and payment method are correlated with each other and with tenure; observed gaps overstate what any single intervention would deliver.
* **Contract effects are partly mechanical** (customers on fixed terms cannot leave freely), so the contract gap should not be read as the achievable effect of an offer.
* **Estimates are simple.** The CLV estimate assumes constant monthly churn hazard and constant ARPU, undiscounted, capped at 60 months; revenue-at-risk figures are exposure measures, not forecasts.
* **Model:** one random split + CV; hyper-parameters not tuned; model differences within noise; probabilities not calibrated; no cost data for threshold choice.
* **Data:** a public, well-known sample dataset; findings illustrate method and may not transfer to another operator. Customer-level explanations (SHAP) and tuning are planned for Phase 2 (see README).
