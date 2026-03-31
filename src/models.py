from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_models():
    models = {
        "Ridge" : Ridge(),
        "RandomForest" : RandomForestRegressor(),
        "XGBoost" : XGBRegressor()
    }

    return models