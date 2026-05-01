from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os
import ast
import warnings
warnings.filterwarnings('ignore')
 
app = Flask(__name__)
CORS(app)
 
# ── Load model files ─────────────────────────────────────────────────
model  = joblib.load("model.pkl")
le     = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
 
with open("top_materials.json", "r") as f:
    TOP_DATA = json.load(f)
 
ALL_CHEMICALS = sorted(le.classes_.tolist())
 
# ── Load and enrich medicine CSV ─────────────────────────────────────
CSV_FILE     = "medicine.csv"
df_medicines = None
chem_stats   = None
name_col = comp_col = None
 
if os.path.exists(CSV_FILE):
    df_medicines = pd.read_csv(CSV_FILE)
    df_medicines.columns = df_medicines.columns.str.strip()
    name_col = "brand_name"
    comp_col = "active_ingredients"
    df_medicines["_name_lower"] = df_medicines["brand_name"].astype(str).str.lower().str.strip()
 
    # Build chemical risk stats from real data
    chem_stats = df_medicines.groupby("primary_ingredient").agg(
        total_medicines  = ("brand_name",      "count"),
        discontinued_count = ("is_discontinued","sum"),
        manufacturer_count = ("manufacturer",   "nunique"),
        avg_price        = ("price_inr",        "mean"),
        median_price     = ("price_inr",        "median"),
        dosage_forms     = ("dosage_form",      "nunique")
    ).reset_index()
 
    chem_stats["discontinued_rate"] = (
        chem_stats["discontinued_count"] / chem_stats["total_medicines"] * 100
    ).round(1)
    chem_stats["confidence"] = (100 - chem_stats["discontinued_rate"]).clip(0,100).round(1)
 
    def shortage_risk(row):
        if row["discontinued_rate"] > 3 or row["total_medicines"] > 5000:
            return "High Risk"
        elif row["discontinued_rate"] > 1.5 or row["total_medicines"] > 2000:
            return "Medium Risk"
        return "Low Risk"
 
    chem_stats["shortage_risk"] = chem_stats.apply(shortage_risk, axis=1)
    print(f"CSV loaded: {len(df_medicines)} rows")
else:
    print("WARNING: medicine.csv not found")
 
 
# ── Helpers ──────────────────────────────────────────────────────────
 
def parse_ingredients(val):
    try:
        items = ast.literal_eval(str(val))
        return [{"name": i["name"], "strength": i.get("strength",""), "desc": i.get("full_description","")} for i in items]
    except:
        return []
 
def predict_demand(name):
    name = name.lower().strip()
    if name not in le.classes_:
        return None
    encoded    = le.transform([name])
    encoded_df = pd.DataFrame(encoded, columns=["chemical_encoded"])
    scaled     = scaler.transform(encoded_df)
    return float(round(model.predict(scaled)[0], 1))
 
def demand_level(score):
    if score >= 5000: return "Very High"
    if score >= 2500: return "High"
    if score >= 1000: return "Medium"
    return "Low"
 
def get_chem_risk(chemical_name):
    if chem_stats is None:
        return {"shortage_risk": "Unknown", "confidence": 0}
    row = chem_stats[chem_stats["primary_ingredient"].str.lower() == chemical_name.lower()]
    if row.empty:
        return {"shortage_risk": "Unknown", "confidence": 85.0}
    r = row.iloc[0]
    return {
        "shortage_risk":      r["shortage_risk"],
        "confidence":         float(r["confidence"]),
        "discontinued_rate":  float(r["discontinued_rate"]),
        "manufacturer_count": int(r["manufacturer_count"]),
        "avg_price":          round(float(r["avg_price"]),1),
        "median_price":       round(float(r["median_price"]),1)
    }
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 1 — Predict demand  (ENHANCED with risk + confidence)
# ══════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    chemical = request.json.get("chemical","").strip()
    if not chemical:
        return jsonify({"error": "No chemical name provided"}), 400
 
    result = predict_demand(chemical)
    if result is None:
        return jsonify({"error": f"'{chemical}' not found. Try: aceclofenac, cefixime, amoxycillin, azithromycin"}), 404
 
    risk = get_chem_risk(chemical)
 
    return jsonify({
        "chemical":         chemical.lower(),
        "predicted_demand": result,
        "demand_level":     demand_level(result),
        "shortage_risk":    risk.get("shortage_risk","Unknown"),
        "confidence":       risk.get("confidence", 85.0),
        "discontinued_rate":risk.get("discontinued_rate", 0),
        "manufacturer_count": risk.get("manufacturer_count", 0),
        "avg_price":        risk.get("avg_price", 0),
        "insights": _build_insights(chemical, result, risk)
    })
 
def _build_insights(chemical, demand, risk):
    insights = []
    lvl = demand_level(demand)
    if lvl in ("Very High","High"):
        insights.append(f"High usage across {int(demand)} medicines — stock this chemical in advance.")
    if risk.get("discontinued_rate",0) > 2:
        insights.append(f"{risk['discontinued_rate']}% of medicines using this are discontinued — watch for supply risk.")
    mfr = risk.get("manufacturer_count",0)
    if mfr > 0:
        if mfr < 500:
            insights.append(f"Only {mfr} manufacturers produce this — limited supplier options increases shortage risk.")
        else:
            insights.append(f"{mfr} manufacturers produce this — good supplier diversity, lower shortage risk.")
    return insights
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 2 — Compare two chemicals
# ══════════════════════════════════════════════════════════════════════
@app.route("/compare", methods=["POST"])
def compare():
    c1 = request.json.get("chemical1","").strip()
    c2 = request.json.get("chemical2","").strip()
    if not c1 or not c2:
        return jsonify({"error": "Provide both chemical names"}), 400
    r1 = predict_demand(c1)
    r2 = predict_demand(c2)
    if r1 is None: return jsonify({"error": f"'{c1}' not found"}), 404
    if r2 is None: return jsonify({"error": f"'{c2}' not found"}), 404
    rsk1 = get_chem_risk(c1)
    rsk2 = get_chem_risk(c2)
    return jsonify({
        "chemical1": {"name":c1.lower(),"demand":r1,"level":demand_level(r1),
                      "shortage_risk":rsk1.get("shortage_risk","Unknown"),
                      "confidence":rsk1.get("confidence",85),
                      "manufacturer_count":rsk1.get("manufacturer_count",0)},
        "chemical2": {"name":c2.lower(),"demand":r2,"level":demand_level(r2),
                      "shortage_risk":rsk2.get("shortage_risk","Unknown"),
                      "confidence":rsk2.get("confidence",85),
                      "manufacturer_count":rsk2.get("manufacturer_count",0)},
        "higher_demand": c1.lower() if r1 >= r2 else c2.lower()
    })
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 3 — Top 5 / Bottom 5
# ══════════════════════════════════════════════════════════════════════
@app.route("/top-bottom", methods=["GET"])
def top_bottom():
    sd = sorted(TOP_DATA, key=lambda x: x["demand_count"], reverse=True)
    return jsonify({"top5": sd[:5], "bottom5": sd[-5:]})
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 4 — Top materials (enhanced with risk data)
# ══════════════════════════════════════════════════════════════════════
@app.route("/top-materials", methods=["GET"])
def top_materials():
    enhanced = []
    for item in TOP_DATA:
        risk = get_chem_risk(item["chemical"])
        enhanced.append({**item, **risk})
    return jsonify(enhanced)
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 5 — Autocomplete
# ══════════════════════════════════════════════════════════════════════
@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q","").lower().strip()
    if not q: return jsonify([])
    matches = [c for c in ALL_CHEMICALS if q in c.lower()][:8]
    return jsonify(matches)
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 6 — Medicine → full detail (ingredients WITH strength)
# ══════════════════════════════════════════════════════════════════════
@app.route("/medicine-lookup", methods=["POST"])
def medicine_lookup():
    if df_medicines is None:
        return jsonify({"error": "medicine.csv not found"}), 500
    medicine = request.json.get("medicine","").strip().lower()
    if not medicine:
        return jsonify({"error": "No medicine name provided"}), 400
 
    mask    = df_medicines["_name_lower"].str.contains(medicine, na=False)
    matches = df_medicines[mask]
    if matches.empty:
        return jsonify({"error": f"No medicine found matching '{medicine}'"}), 404
 
    results = []
    for _, row in matches.head(10).iterrows():
        ingredients = parse_ingredients(row["active_ingredients"])
        risk        = get_chem_risk(str(row.get("primary_ingredient","")))
        results.append({
            "medicine":         str(row["brand_name"]),
            "manufacturer":     str(row.get("manufacturer","")),
            "price_inr":        round(float(row.get("price_inr",0)),2),
            "dosage_form":      str(row.get("dosage_form","")),
            "pack":             str(row.get("packaging_raw","")),
            "therapeutic_class":str(row.get("therapeutic_class","")),
            "is_discontinued":  bool(row.get("is_discontinued",False)),
            "num_ingredients":  int(row.get("num_active_ingredients",0)),
            "ingredients":      ingredients,
            "primary_ingredient":str(row.get("primary_ingredient","")),
            "shortage_risk":    risk.get("shortage_risk","Unknown"),
            "confidence":       risk.get("confidence",85)
        })
    return jsonify({"query": medicine, "results": results, "count": len(results)})
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 7 — Ingredient → medicines
# ══════════════════════════════════════════════════════════════════════
@app.route("/ingredient-lookup", methods=["POST"])
def ingredient_lookup():
    if df_medicines is None:
        return jsonify({"error": "medicine.csv not found"}), 500
    ingredient = request.json.get("ingredient","").strip().lower()
    if not ingredient:
        return jsonify({"error": "No ingredient provided"}), 400
 
    mask    = df_medicines["active_ingredients"].astype(str).str.lower().str.contains(ingredient, na=False)
    matches = df_medicines[mask]
    if matches.empty:
        return jsonify({"error": f"No medicines found containing '{ingredient}'"}), 404
 
    medicine_list = matches[["brand_name","manufacturer","price_inr","dosage_form","therapeutic_class","is_discontinued"]].head(30).to_dict("records")
    risk = get_chem_risk(ingredient)
 
    return jsonify({
        "ingredient":       ingredient,
        "medicines":        medicine_list,
        "total":            len(matches),
        "shortage_risk":    risk.get("shortage_risk","Unknown"),
        "confidence":       risk.get("confidence",85),
        "manufacturer_count": risk.get("manufacturer_count",0),
        "avg_price":        risk.get("avg_price",0)
    })
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 8 — Dashboard summary stats
# ══════════════════════════════════════════════════════════════════════
@app.route("/dashboard-stats", methods=["GET"])
def dashboard_stats():
    if df_medicines is None:
        return jsonify({"error": "CSV not loaded"}), 500
    total       = int(len(df_medicines))
    discontinued= int(df_medicines["is_discontinued"].sum())
    active      = total - discontinued
    manufacturers = int(df_medicines["manufacturer"].nunique())
    chemicals   = int(df_medicines["primary_ingredient"].nunique())
    high_risk   = int(chem_stats[chem_stats["shortage_risk"]=="High Risk"]["primary_ingredient"].count()) if chem_stats is not None else 0
    by_class    = df_medicines["therapeutic_class"].value_counts().head(8).to_dict()
    by_form     = df_medicines["dosage_form"].value_counts().head(6).to_dict()
    return jsonify({
        "total_medicines":    total,
        "active_medicines":   active,
        "discontinued":       discontinued,
        "manufacturers":      manufacturers,
        "unique_chemicals":   chemicals,
        "high_risk_chemicals":high_risk,
        "by_therapeutic_class": by_class,
        "by_dosage_form":     by_form
    })
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 9 — Shortage risk report
# ══════════════════════════════════════════════════════════════════════
@app.route("/shortage-risk", methods=["GET"])
def shortage_risk_report():
    if chem_stats is None:
        return jsonify({"error": "CSV not loaded"}), 500
    top_risk = chem_stats.sort_values("total_medicines", ascending=False).head(1586)
    result   = []
    for _, r in top_risk.iterrows():
        result.append({
            "chemical":           str(r["primary_ingredient"]),
            "total_medicines":    int(r["total_medicines"]),
            "discontinued_count": int(r["discontinued_count"]),
            "discontinued_rate":  float(r["discontinued_rate"]),
            "manufacturer_count": int(r["manufacturer_count"]),
            "avg_price":          round(float(r["avg_price"]),1),
            "shortage_risk":      str(r["shortage_risk"]),
            "confidence":         float(r["confidence"])
        })
    return jsonify(result)
 
 
# ══════════════════════════════════════════════════════════════════════
# ROUTE 10 — Health
# ══════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "running",
        "model":       "RandomForestRegressor",
        "csv_loaded":  df_medicines is not None,
        "chemicals":   len(ALL_CHEMICALS),
        "total_meds":  int(len(df_medicines)) if df_medicines is not None else 0
    })
 
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)