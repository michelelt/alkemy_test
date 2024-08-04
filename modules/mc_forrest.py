from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error



def mc_forrest(X_train, y_train, X_test, y_test):
    # Crea il modello Random Forest
    clf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Allenamento del modello
    clf.fit(X_train, y_train)

    # Predizioni sui dati di test
    y_pred = clf.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {'mape':mape, 'mae':mae, 'model':clf}
