import mlflow

def prediction(data, model_path):
    print('start prediction')
    loaded_model = mlflow.pyfunc.load_model(model_path, suppress_warnings=True)
    result = loaded_model.predict(data)
    return result[0]
