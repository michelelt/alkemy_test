import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


# Definire una funzione di gradiente e hessiano per MAPE
def mape_obj(preds, dtrain):
    labels = dtrain.get_label()
    d = preds - labels
    grad = np.where(labels != 0, np.sign(d) / labels, 0.0)
    hess = np.where(labels != 0, 1.0 / np.abs(labels), 0.0)
    return grad, hess

def mc_xgb(X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Impostare i parametri di base di XGBoost
    params = {
        'max_depth': 5,
        'eta': 0.1,
        'objective': 'reg:squarederror'
    }

    # Addestramento del modello con la funzione di perdita personalizzata
    model = xgb.train(params, dtrain, num_boost_round=100,)

    # Valutare le prestazioni del modello
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)

    mape = mean_absolute_percentage_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return {'mape':mape, 'mae':mae, 'model':model}