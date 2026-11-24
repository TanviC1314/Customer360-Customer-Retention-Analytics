# Raw data

`WA_Fn-UseC_-Telco-Customer-Churn.csv` - the IBM Telco Customer Churn sample dataset, **unmodified**.

| Item | Value |
|---|---|
| Rows / columns | 7,043 customers x 21 columns |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Downloaded from | IBM's public GitHub sample-data repo: <https://github.com/IBM/telco-customer-churn-on-icp4d> (`data/Telco-Customer-Churn.csv`), saved under the Kaggle file name |
| Original listing | <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |

The Kaggle download needs a login, so the file was fetched from IBM's public mirror of the same sample data.
It is the same dataset (7,043 rows, 21 columns, identical schema). It has not been byte-compared with the Kaggle
download, so if you want to be certain, download it from Kaggle manually (below) and compare the SHA-256.

## Manual download (alternative)

1. Sign in at <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> and click **Download**.
2. Unzip `WA_Fn-UseC_-Telco-Customer-Churn.csv` into this folder.
3. Verify: `Get-FileHash data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv` (PowerShell).

Or via the Kaggle CLI: `kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip`

**Never edit this file.** All cleaning happens in code and writes to `data/processed/`.
