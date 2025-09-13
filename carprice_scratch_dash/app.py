# ======================
# 1) Imports & artifacts
# ======================
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State

APP_DIR = Path(__file__).resolve().parent
ART_S   = APP_DIR / "artifacts_scratch"
ART_RF  = APP_DIR / "artifacts_rf"

rf_pipe   = joblib.load(ART_RF / "RandomForestRegressor.z")
preproc_s = joblib.load(ART_S  / "preproc.pkl")
theta_s   = np.load(ART_S / "theta.npy")
meta_s    = json.loads((ART_S / "meta.json").read_text())

CAT_COLS   = meta_s["cat_cols"]
NUM_COLS   = meta_s["num_cols"]
LOG_TARGET = meta_s["log_target"]
INTERCEPT  = meta_s["intercept_in_theta"]

# OHE cats (for dropdowns)
try:
    ohe = preproc_s.named_transformers_["cat"].named_steps["ohe"]
    OHE_CATS = {c: list(cat) for c, cat in zip(CAT_COLS, ohe.categories_)}
except Exception:
    OHE_CATS = {
        "fuel": ["Diesel","Petrol","CNG","LPG","Electric"],
        "seller_type": ["Individual","Dealer","Trustmark Dealer"],
        "transmission": ["Manual","Automatic"],
        "brand": []
    }

# ======================
# 2) Predict helpers + UI helpers
# ======================
def _add_intercept(X): 
    return np.hstack([np.ones((X.shape[0],1)), X])

def predict_scratch(row: dict) -> float:
    df = pd.DataFrame([row], columns=NUM_COLS + CAT_COLS)
    X  = preproc_s.transform(df)
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X  = _add_intercept(X) if INTERCEPT else X
    y_log = X @ theta_s
    y = np.exp(np.clip(y_log, -50, 50)) if LOG_TARGET else y_log
    return float(y[0])

def predict_rf(row: dict) -> float:
    A1_NUM = ["engine","max_power","mileage","seats","year","km_driven","owner"]
    A1_CAT = ["fuel","seller_type","transmission","brand"]
    df_raw = pd.DataFrame([row], columns=A1_NUM + A1_CAT)
    return float(rf_pipe.predict(df_raw).ravel()[0])

def num_input(id_, label, step='0.1', required=False, value=None):
    return html.Div([ html.Label(label),
        dcc.Input(id=id_, type="number", step=step, debounce=True,
                  required=required, value=value, style={"width":"100%"})
    ], style={"marginBottom":"8px"})

def dd(id_, label, options, value=None):
    opts = [{"label": str(x), "value": str(x)} for x in options] if options else []
    return html.Div([ html.Label(label),
        dcc.Dropdown(id=id_, options=opts, value=value,
                     placeholder=f"Select {label}", searchable=True, style={"width":"100%"})
    ], style={"marginBottom":"8px"})

def shared_inputs():
    return html.Div([
        html.Div([
            html.H4("Numeric features"),
            num_input("engine","engine (cc)"),
            num_input("max_power","max_power (bhp)"),
            num_input("mileage","mileage (kmpl)"),
            num_input("seats","seats", step="1"),
            num_input("year","year", step="1", required=True),
            num_input("km_driven","km_driven", step="1", required=True),
            num_input("owner","owner (0..n)", step="1", required=True),
        ], style={"flex":"1","minWidth":"260px","paddingRight":"16px"}),
        html.Div([
            html.H4("Categorical features"),
            dd("fuel","fuel", OHE_CATS.get("fuel", []),
               (OHE_CATS.get("fuel",[None])[0] if OHE_CATS.get("fuel") else None)),
            dd("seller_type","seller_type", OHE_CATS.get("seller_type", [])),
            dd("transmission","transmission", OHE_CATS.get("transmission", [])),
            dd("brand","brand", OHE_CATS.get("brand", [])),
        ], style={"flex":"1","minWidth":"260px","paddingLeft":"16px"}),
    ], style={"display":"flex","gap":"16px","flexWrap":"wrap"})

def nav_bar(active="/"):
    def link(href,label):
        style={"padding":"10px 14px","textDecoration":"none","color":"#333","borderBottom":"3px solid transparent"}
        if href==active: 
            style["borderBottom"]="3px solid #0070f3"
            style["fontWeight"]="600"
        return html.A(label, href=href, style=style)

    # greeting only on home
    greeting = html.Div("👋 Welcome, st126055", style={"fontWeight":"500","color":"#555"}) if active=="/" else None

    return html.Div([
        html.Div("Car Price Predictor", style={"fontSize":"20px","fontWeight":"700"}),
        html.Div([link("/", "Home"), link("/old","Old Model (A1)"), link("/new","New Model (A2)")],
                 style={"display":"flex","gap":"8px"}),
        greeting
    ], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
              "padding":"10px 16px","borderBottom":"1px solid #eee","position":"sticky",
              "top":0,"background":"#fff","zIndex":999})

def card(children, title=None, color="#fff"):
    return html.Div([
        html.Div(title, style={"fontWeight":"600","marginBottom":"8px"}) if title else None,
        children
    ], style={"background":color,"padding":"14px","border":"1px solid #eee","borderRadius":"12px"})

def collect_row(engine,max_power,mileage,seats,year,km_driven,owner,fuel,seller_type,transmission,brand):
    return dict(engine=engine,max_power=max_power,mileage=mileage,seats=seats,
                year=year,km_driven=km_driven,owner=owner,
                fuel=fuel,seller_type=seller_type,transmission=transmission,brand=str(brand) if brand is not None else None)

# ======================
# 3) PAGE FUNCTIONS
# ======================
def page_home():
    return html.Div([
        html.H1("🚗 Welcome to st126055 Car Price Predictor",
                style={"textAlign":"center", "color":"#2c3e50"}),

        html.H4("Check your car’s price using two different models",
                style={"textAlign":"center", "color":"#7f8c8d"}),

        card(dcc.Markdown(
            """
            ### How to use
            1. Go to **Old Model (A1)** → uses RandomForest (baseline model).  
            2. Go to **New Model (A2)** → uses Linear Regression from scratch (our improved model).  
            3. Fill in your car’s details (year, engine, mileage, etc.).  
            4. Click **Predict** → get the estimated selling price in Thai Baht.  
            """
        ), title="Steps"),

        card(dcc.Markdown(
            """
            ### What’s different between the models?
            - **A1: Old Model (RandomForest)**  
              - Baseline model trained in Assignment 1.  
              - Non-linear, black-box model.  
              - Works well, but harder to interpret.  

            - **A2: New Model (Scratch Linear Regression)**  
              - Built from scratch using gradient descent, momentum, and Xavier initialization.  
              - Regularization (Ridge/Lasso) improves stability.  
              - Trained on log(price) → predictions are smoother.  
              - Easier to explain coefficients (feature importance).  
            """
        ), title="Model Comparison", color="#eef6ff"),

        card(dcc.Markdown(
            """
            ### Why this matters
            By comparing the two models, we can see how a **baseline ML algorithm (RF)** differs from a
            **custom, interpretable model** built step-by-step in A2. This shows both performance and explainability.  
            """
        ), title="Takeaway", color="#f9f9f9")
    ], style={"maxWidth":"850px","margin":"0 auto","padding":"20px"})

def page_old():
    return html.Div([
        html.H2("🌲 Old Model (A1) — RandomForest", style={"color":"#d35400"}),
        card("This is the **old model** from A1.\nIt uses **RandomForest**. Good baseline, but harder to explain.",
             title="About this model", color="#fff6ef"),
        shared_inputs(),
        html.Div([
            html.Button("Fill sample", id="btn_sample", n_clicks=0,
                        style={"marginRight":"10px","background":"#f39c12","color":"#fff"}),
            html.Button("Predict (A1)", id="btn_predict_a1", n_clicks=0,
                        style={"background":"#e67e22","color":"#fff"}),
            html.Button(id="btn_sample", n_clicks=0, style={"display": "none"}) 
        ], style={"margin":"12px 0"}),
        dcc.Loading(children=html.Div(id="pred_wrap_a1")),
        html.Small(id="debug_msg_a1", style={"color":"#666"})
    ], style={"maxWidth":"850px","margin":"0 auto","padding":"20px"})

def page_new():
    return html.Div([
        html.H2("🌟 New Model (A2) — Scratch Linear Regression", style={"color":"#27ae60"}),
        card("This is the **new model** from A2.\nIt uses **Linear Regression from scratch**. Clearer, more stable, and better explained.",
             title="About this model", color="#ecfff3"),
        shared_inputs(),
        html.Div([
            html.Button("Fill sample", id="btn_sample", n_clicks=0,
            style={"marginRight":"10px","background":"#2ecc71","color":"#fff"}),
            html.Button("Predict (A2)", id="btn_predict_a2", n_clicks=0,
                        style={"background":"#27ae60","color":"#fff"}),
            html.Button(id="btn_sample_new", n_clicks=0, style={"display": "none"})
        ], style={"margin":"12px 0"}),
        dcc.Loading(children=html.Div(id="pred_wrap_a2")),
        html.Small(id="debug_msg_a2", style={"color":"#666"})
    ], style={"maxWidth":"850px","margin":"0 auto","padding":"20px"})

# ======================
# 4) APP + ROUTER
# ======================
app = Dash(__name__, title="Car Price Predictor — A1 & A2", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="navbar"),
    html.Div(id="page-container", style={"maxWidth":"1000px","margin":"24px auto","padding":"0 16px"})
])

app.validation_layout = html.Div([
    app.layout,
    nav_bar("/"),
    page_home(),
    page_old(),
    page_new(),
])

@app.callback(Output("navbar","children"), Output("page-container","children"), Input("url","pathname"))
def router(pathname):
    nav = nav_bar(active=pathname or "/")
    if pathname in ("/", None): return nav, page_home()
    if pathname == "/old":     return nav, page_old()
    if pathname == "/new":     return nav, page_new()
    return nav, html.Div("Page not found", style={"padding":"24px"})

# ======================
# 5) CALLBACKS
# ======================
@app.callback(
    Output("engine","value"), Output("max_power","value"), Output("mileage","value"), Output("seats","value"),
    Output("year","value"), Output("km_driven","value"), Output("owner","value"),
    Output("fuel","value"), Output("seller_type","value"), Output("transmission","value"), Output("brand","value"),
    Input("btn_sample","n_clicks"),
    prevent_initial_call=True
)
def fill_sample(n):
    return 1496, 110, 19.5, 5, 2018, 40000, 1, "Diesel", "Dealer", "Manual", "Toyota"

@app.callback(
    Output("pred_wrap_a1","children"), Output("debug_msg_a1","children"),
    Input("btn_predict_a1","n_clicks"),
    State("engine","value"), State("max_power","value"), State("mileage","value"), State("seats","value"),
    State("year","value"), State("km_driven","value"), State("owner","value"),
    State("fuel","value"), State("seller_type","value"), State("transmission","value"), State("brand","value"),
    prevent_initial_call=True
)
def on_predict_a1(n, engine, max_power, mileage, seats, year, km_driven, owner, fuel, seller_type, transmission, brand):
    try:
        row = collect_row(engine,max_power,mileage,seats,year,km_driven,owner,fuel,seller_type,transmission,brand)
        price = predict_rf(row)
        return card(html.Div(f"Predicted price: {price:,.0f} ฿",
                     style={"fontSize":"28px","fontWeight":"700","color":"#27ae60"}),title="Predicted price"), "[OK] RF pipeline"
    except Exception as e:
        return card("Prediction failed.", title="Error", color="#fff5f5"), f"{type(e).__name__}: {e}"

@app.callback(
    Output("pred_wrap_a2","children"), Output("debug_msg_a2","children"),
    Input("btn_predict_a2","n_clicks"),
    State("engine","value"), State("max_power","value"), State("mileage","value"), State("seats","value"),
    State("year","value"), State("km_driven","value"), State("owner","value"),
    State("fuel","value"), State("seller_type","value"), State("transmission","value"), State("brand","value"),
    prevent_initial_call=True
)
def on_predict_a2(n, engine, max_power, mileage, seats, year, km_driven, owner, fuel, seller_type, transmission, brand):
    try:
        row = collect_row(engine,max_power,mileage,seats,year,km_driven,owner,fuel,seller_type,transmission,brand)
        price = predict_scratch(row)
        return card(html.Div(f"Predicted price: {price:,.0f} ฿",
                     style={"fontSize":"32px","fontWeight":"800","color":"#27ae60"}),
            title="Predicted price"), "[OK] Scratch model"
    except Exception as e:
        return card("Prediction failed.", title="Error", color="#fff5f5"), f"{type(e).__name__}: {e}"

@server.route("/health")
def _health(): return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)