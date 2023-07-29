import lightgbm as lgb


def prediction(data, model_path):
    bst = lgb.Booster(model_file=model_path)
    result = bst.predict(data)
    return result[0]
