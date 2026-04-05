import shap

def explain_prediction(pipeline, x_sample):
    model = list(pipeline.named_steps.values())[-1]
    preprocessor = pipeline.named_steps["preprocessing"]

    #input transformation
    x_transformed = preprocessor.transform(x_sample)

    explainer = shap.Explainer(model)

    shap_values = explainer(x_transformed)

    return shap_values