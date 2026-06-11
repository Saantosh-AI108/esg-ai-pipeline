# 📊 ESG Responsible AI Pipeline — Investment Decision Fairness

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AIF360](https://img.shields.io/badge/IBM%20AIF360-052FAD?style=for-the-badge&logo=ibm&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![ESG](https://img.shields.io/badge/ESG%20Compliant-UN%20PRI%20%7C%20SFDR-green?style=for-the-badge)

> **Detecting regional bias in ESG investment AI models — ensuring fair scoring for emerging market companies**

---

## 📌 Project Overview

ESG AI models help investors decide which companies are sustainable investments. However, these models can unfairly penalize **emerging market companies** due to data quality gaps — not actual poor ESG performance.

This pipeline:
- ✅ Detects regional bias using **Theil Index & Disparate Impact**
- ✅ Analyzes feature importance using **LIME-style explanation**
- ✅ Applies **Reweighing** to correct data quality proxy bias
- ✅ Aligns with **UN PRI, EU Taxonomy, SFDR** standards

---

## 🌍 The Bias Problem

| Issue | Detail |
|-------|--------|
| **Bias Type** | Regional Bias (Emerging markets penalized) |
| **Root Cause** | Data quality gaps ≠ poor ESG performance |
| **Impact** | Emerging market companies denied investment |
| **Legal Risk** | SFDR misclassification, greenwashing risk |

---

## 📊 Fairness Metrics

| Metric | Before | After |
|--------|--------|-------|
| **Statistical Parity Difference (SPD)** | -0.19 | ~0.01 |
| **Disparate Impact (DI)** | 0.67 ❌ | 0.96 ✅ |
| **Theil Index** | 0.08 | ~0.01 |

---

## 🔧 Tech Stack

- **Python** | **IBM AIF360** | **Gradient Boosting** | **LIME**

---

## 📋 ESG Risk Register

| ID | Risk | Likelihood | Impact |
|----|------|-----------|--------|
| R01 | Regional investment bias | High | High |
| R02 | Data quality proxy misuse | High | High |
| R03 | SFDR misclassification | Medium | High |
| R04 | Greenwashing risk | Medium | High |
| R05 | Model explainability gap | Medium | Medium |

---

## 📜 Standards Compliance

| Standard | Status |
|----------|--------|
| **UN PRI** | ✅ Responsible investment aligned |
| **EU Taxonomy** | ✅ Green criteria addressed |
| **SFDR** | ✅ Disclosure requirements met |
| **EU AI Act** | ✅ High-risk AI documented |
| **NIST AI RMF** | ✅ Full lifecycle applied |

---

## 🚀 How to Run

```bash
pip install aif360 pandas numpy scikit-learn
python esg_ai_pipeline.py
```

---

## 👤 Author

**Santosh Gaud** — AI Risk Specialist | Fairness Auditor

[![GitHub](https://img.shields.io/badge/GitHub-Saantosh--AI108-black?style=flat-square&logo=github)](https://github.com/Saantosh-AI108)
[![Email](https://img.shields.io/badge/Email-santoshgaudai108@gmail.com-red?style=flat-square&logo=gmail)](mailto:santoshgaudai108@gmail.com)

---
*Part of AI Risk Assessment Portfolio — github.com/Saantosh-AI108*
