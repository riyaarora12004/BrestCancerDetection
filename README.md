# Breast Cancer Risk Screening Helper

A machine learning desktop app that estimates whether a breast lump is likely to be benign (harmless) or malignant (cancerous) from biopsy cell measurements. MCA mini project, Chandigarh University.

> **Disclaimer:** Educational project only. Not a medical device. If you notice a lump or any change, consult a doctor.

## Features
- Two models: Logistic Regression and Random Forest (200 trees)
- Six easy inputs with plain-language explanations
- Verdict: Low Risk / Uncertain / High Risk, with risk percentage from each model
- Chart showing which inputs raise or lower the risk
- "How it works & accuracy" tab in simple language

## Dataset
Wisconsin Diagnostic Breast Cancer (569 patients), built into scikit-learn. No download needed.

## Results (143 unseen test patients)
| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Test accuracy | 94.4% | 95.1% |
| Cancers detected | 48 of 53 | 51 of 53 |
| Cancers missed | 5 | 2 |
| False alarms | 3 | 5 |

## Run
```bash
pip install -r requirements.txt
python cancer_risk_helper.py
```
Requires Python 3.9+ (Tkinter included).

## Author
Riya Arora, UID 25MCA20138, Section 25MCA1B. Guide: Ms. Reeti Jaswal.
