from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using MAE, MSE, and R-squared metrics.

    Parameters:
        y_true (array-like): The ground truth target values.
        y_pred (array-like): The predicted values from the model.

    Returns:
        dict: A dictionary containing the MAE, MSE, and R-squared values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-squared': r2
    }
