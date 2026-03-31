def get_feature_importance(pipeline):
    model = list(pipeline.named_steps.values())[-1]
    preprocessor = pipeline.named_steps["preprocessing"]

    feature_names = preprocessor.get_feature_names_out()

    #Tree based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        feature_importance = sorted(
            zip(feature_names, importances),
            key = lambda x: x[1],
            reverse=True
        )

        return feature_importance
    
    #linear models
    elif hasattr(model, "coef_"):
        importances = model.coef_

        feature_importance = sorted(
            zip(feature_names, importances),
            key = lambda x: abs(x[1]),
            reverse=True
        )

        return feature_importance
    
    else:
        return None
