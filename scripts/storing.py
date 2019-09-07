import joblib

def save_model(model, path):
    joblib.dump(model, path, compress=1)
    print('Model was saved in', path)
    print('Use load_model to load model from file.')

def load_model(path):
    model = joblib.load(path)
    print('Model was loaded from', path)
    return model