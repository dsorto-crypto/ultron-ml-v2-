import lightgbm as lgb
import joblib

print("Loading LightGBM booster from ultron_v3_entry_lgbm.txt...")

model = lgb.Booster(model_file="ultron_v3_entry_lgbm.txt")

joblib.dump(model, "model.pkl")

print("Converted ultron_v3_entry_lgbm.txt → model.pkl successfully.")
