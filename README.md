# A2: Car Price Prediction (st126055)

This is my Assignment 2 for Machine Learning class.  
The goal is to build and compare two models for predicting car price:

- **Old Model (A1):** RandomForest baseline  
- **New Model (A2):** Linear Regression from scratch (with Xavier init, momentum, and regularization)

Both models are also deployed in a Dash web app.

---

## Files in this repo
- `A2_st126055_CarPrice.ipynb` — Jupyter notebook (main work)  
- `A2_st126055_CarPrice.pdf` — exported report version  
- `carprice_scratch_dash/` — Dash web app folder  
  - `app.py` — web application code  
  - `Dockerfile`, `docker-compose.yaml` — for containerization & deployment  
  - `artifacts/` — preprocessor and model weights  
- `images/` — plots and screenshots for report

---

## How to run locally
```bash
cd carprice_scratch_dash
docker build -t carprice_app .
docker run -p 8050:8050 carprice_app
