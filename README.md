# 📊 Academic Data Diagnosis Helper

A guided regression diagnosis tool designed for thesis analysis.  
It helps users identify data issues, validate model assumptions, and iteratively improve regression models.

---

## 🚀 Background

In thesis empirical analysis, students often struggle with:

- Detecting outliers and handling extreme values  
- Identifying multicollinearity  
- Missing nonlinear relationships or interaction effects  
- Understanding why regression results are not significant  

This tool provides a **step-by-step diagnostic workflow** to systematically address these issues.

---

## 🧩 Features

### Step 1: Outlier Detection
- Detect extreme values (including dependent & independent variables)
- Support winsorization (1%–99%)
- Identify long-tail distributions

### Step 2: Multicollinearity Check
- Compute VIF for all variables  
- Classify severity:
  - VIF < 5 → No issue  
  - 5–10 → Moderate  
  - >10 → Severe  

### Step 3: Nonlinearity & Interaction
- Detect nonlinear relationships  
- Suggest interaction terms  
- Allow users to include terms in model  

### Step 4: Model Refinement
- Iteratively improve regression specification  
- Compare before vs after  

### Step 5: Final Model Output
- Final regression results  
- Initial vs final comparison (R² / Adj R²)  
- Coefficient comparison  

---

## 📈 Example Output

- Clear regression tables  
- Highlighted insignificant variables  
- Step-by-step improvement tracking  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Statsmodels  
- Pandas  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py<img width="1897" height="1187" alt="image" src="https://github.com/user-attachments/assets/6364b96f-998e-4f58-9c7d-5d484bd7ae38" />
