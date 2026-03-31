import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessor import SmartPreprocessor
from src.models import get_models

#loading the data
data = pd.read_csv("data/car data.csv")
target_column = "Selling_Price"

#features
data = data.dropna(subset=[target_column])

x = data.drop(target_column, axis=1)
x = x.drop("Car_Name", axis=1)
y = data[target_column]

#split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#get the models
models = get_models()

results = {}
trained_pipelines = {}

#train models
for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", SmartPreprocessor()),
        ("model", model)
    ])

    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {
        "MSE" : mse,
        "R2" : r2
    }

    trained_pipelines[name] = pipeline

#results
print("\nModel Comparison Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: R2 = {metrics['R2']:.4f}, MSE = {metrics['MSE']:.4f}")


#select the best model
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_pipeline = trained_pipelines[best_model_name]

print(f"\nBest Model : {best_model_name}")

#feature names
feature_names = best_pipeline.named_steps["preprocessing"].get_feature_names_out()
print("\nFinal Features used:")
print(feature_names)

from src.explain import get_feature_importance

print("\nFeature Importance:")

importance = get_feature_importance(best_pipeline)

if importance:
    for feature, score in importance:
        print(f"{feature}: {score:.4f}")