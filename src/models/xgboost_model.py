from dataclasses import dataclass
from typing import Optional

import numpy as np
from xgboost import XGBRegressor

@dataclass
class XGBoostConfig:
    """
    Configuration for XGBoost model.
    
    This model is designed for tabular factor features.

    It predicts future stock returns or style-adjusted returns.

    @JulongQuant
    """

    # Model parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1

class XGBoostReturnRegressor:
    """
    XGBoost wrapper for stock return regression.

    Input:
        X: 2D numpy array with shape [n_samples, n_features]

    Target:
        y: 1D numpy array with shape [n_samples]

    Output:
        prediction: 1D numpy array with shape [n_samples]
    """

    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self._is_fitted = False
        self.model = XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            objective='reg:squarederror',
        )
    
    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_valid: Optional[np.ndarray] = None,
            y_valid: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the XGBoost model.
        """

        # Flatten y to ensure 1D shape
        y_train = np.ravel(y_train)
        
        self._check_X_y(X_train, y_train)

        if X_valid is not None and y_valid is not None:
            y_valid = np.ravel(y_valid)
            self._check_X_y(X_valid, y_valid)
        
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        else:
            self.model.fit(
                X_train, 
                y_train, 
                verbose=False
            )
        
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained XGBoost model.
        """

        if X.ndim != 2:
            raise ValueError(f"Expected input shape [n_samples, n_features], but got {X.shape}")
        
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        
        preds = self.model.predict(X)

        return preds
    
    @staticmethod
    def _check_X_y(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected X shape [n_samples, n_features], but got {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"Expected y shape [n_samples], but got {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X and y must match, but got {X.shape[0]} and {y.shape[0]}")
        
def build_xgboost_model(config: XGBoostConfig) -> XGBoostReturnRegressor:
    """
    Helper function for building the XGBoost model from config.
    """

    return XGBoostReturnRegressor(config)

if __name__ == "__main__":
    # Quick smoke test.
    config = XGBoostConfig()
    model = build_xgboost_model(config)

    # Mock data
    X_train = np.random.rand(1000, 36)  # 1000 samples, 36 features
    y_train = np.random.rand(1000)      # 1000 target values

    model.fit(X_train, y_train)
    preds = model.predict(X_train)

    print(preds)