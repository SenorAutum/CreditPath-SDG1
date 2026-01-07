# CreditPath: AI for Financial Inclusion 🌍💳
### Machine Learning Meets UN SDG 1: No Poverty

**Author:** Brandon Mutua Mwanzia
**Tech Stack:** Python, Scikit-Learn, Pandas, Matplotlib

## 📌 Project Overview
CreditPath is a Machine Learning solution designed to predict financial inclusion status among populations in East Africa. By analyzing demographic data (employment, education, location), the model identifies individuals who are likely "unbanked" but economically active, helping fintechs and NGOs target them for financial literacy and micro-credit programs.

## 🎯 SDG Alignment
**Goal 1: No Poverty (Target 1.4)**
*Ensure that all men and women, in particular the poor and the vulnerable, have equal rights to economic resources.*
This model aids in extending financial services to underserved regions.

## ⚙️ How It Works (ML Approach)
1.  **Data Source:** Synthetic demographic data (modeled after the Zindi Financial Inclusion dataset).
2.  **Algorithm:** Random Forest Classifier (Supervised Learning).
3.  **Preprocessing:** One-hot encoding for categorical variables; removal of sensitive gender tags to reduce direct bias.

## 📊 Results
* **Accuracy:** ~XX% (See notebook for live results)
* **Key Insight:** 'Job Type' and 'Education Level' were the strongest predictors of banking status.

## 📷 Screenshots
<img width="927" height="625" alt="image" src="https://github.com/user-attachments/assets/f8131b21-17ff-4a15-ab58-79ff79f36970" />


## ⚖️ Ethical Considerations
While the model removes gender labels, "Job Type" can act as a proxy for gender bias in certain cultures. Future iterations will use "Fairness Constraints" to ensure equal False Positive Rates across demographic groups.
Pitch deck link;
https://www.canva.com/design/DAG9t7pDYug/Qk5lD_Rlp1z4wBLmhXLfpw/view?utm_content=DAG9t7pDYug&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hf6e3c08388
