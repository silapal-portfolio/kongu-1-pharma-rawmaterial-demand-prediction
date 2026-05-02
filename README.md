💊 Pharmaceutical Raw Material Demand Predictor

> An ML-powered web system that analyzes 2,53,973 Indian medicine records  
> to predict which raw chemical ingredients will be needed most —  
> helping pharmaceutical companies avoid shortages and plan inventory.

🖥️ Live Screenshots

Predict Demand

![Predict Demand](https://github.com/user-attachments/assets/2fd33c4b-0670-4132-80c7-55c9b4ec5b30)


Medicine Lookup — Full Ingredient Breakdown

![Medicine Lookup](https://github.com/user-attachments/assets/ce3aa23d-a322-41e6-96c7-56a37f988d84)

Ingredient Lookup — All Medicines Containing It

![Ingredient Lookup](https://github.com/user-attachments/assets/b91b7568-9db0-4ec0-83af-e019cc7768f2)


### Compare Two Chemicals

![Compare Chemicals](https://github.com/user-attachments/assets/baa047de-80b9-40ea-9a40-468fd9c904b1)


### Shortage Risk Report

![Shortage Risk](https://github.com/user-attachments/assets/3a6ea4b3-d126-4dc5-8f43-14dcdce39a44)


### Dashboard

![Dashboard](https://github.com/user-attachments/assets/002b72b1-a44e-496f-8b3e-9b4fb85b531b)

---
## 🎯 What This Project Does

This system solves a real pharmaceutical supply chain problem:
- Companies don't know which raw materials to stock more of
- This leads to **medicine shortages** and **wasted inventory**
- Our system predicts demand and flags shortage risk **before it happens**

---
⚙️ Features

| Feature | Description |
|---|---|
| 🔮 **Predict Demand** | Enter any chemical → get demand score + High/Medium/Low badge + confidence % |
| 💊 **Medicine Lookup** | Type a brand name → see all ingredients with exact strength (e.g. Paracetamol 650mg) |
| 🧪 **Ingredient Lookup** | Type a chemical → see all medicines containing it with price and manufacturer |
| ⚖️ **Compare** | Compare two chemicals side by side with demand bars and shortage risk |
| ⚠️ **Shortage Risk** | Full risk table showing discontinued rate and manufacturer concentration |
| 📊 **Dashboard** | KPI cards + bar chart + doughnut chart + full rankings table |

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Regressor
- **Training:** Google Colab
- **Input Feature:** Chemical name (LabelEncoded + StandardScaled)
- **Output:** Predicted demand score (number of medicines using that chemical)
- **Dataset:** 2,53,973 Indian pharmaceutical records from Kaggle
- **Unique chemicals:** 1,586
- **Manufacturers:** 7,648

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Random Forest) |
| Backend | Python, Flask, Flask-CORS |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Data | Pandas, NumPy |
| Training | Google Colab |
