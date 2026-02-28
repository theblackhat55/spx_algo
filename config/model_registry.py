"""
config/model_registry.py
========================
Central registry of every model in the spx_algo system.

Each entry contains:
  model_class     : str   — dotted import path + class name
  target          : str   — one of: target_high | target_low |
                             target_range | target_direction
  default_params  : dict  — hyperparameters used for initial training
  search_space    : dict  — hyperparameter distributions for tuning
  requires_scaling: bool  — True for linear models; False for tree models
"""

MODEL_REGISTRY: dict[str, dict] = {

    # ── LAYER 1 — High prediction ─────────────────────────────────────────────
    "xgboost_high": {
        "model_class": "xgboost.XGBRegressor",
        "target": "target_high",
        "requires_scaling": False,
        "default_params": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        },
        "search_space": {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "min_child_weight": [1, 3, 5],
        },
    },

    "lightgbm_high": {
        "model_class": "lightgbm.LGBMRegressor",
        "target": "target_high",
        "requires_scaling": False,
        "default_params": {
            "n_estimators": 400,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "search_space": {
            "n_estimators": [200, 400, 600],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_samples": [10, 20, 40],
        },
    },

    "catboost_high": {
        "model_class": "catboost.CatBoostRegressor",
        "target": "target_high",
        "requires_scaling": False,
        "default_params": {
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
        },
        "search_space": {
            "iterations": [200, 400, 600],
            "depth": [4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
        },
    },

    "ridge_high": {
        "model_class": "sklearn.linear_model.Ridge",
        "target": "target_high",
        "requires_scaling": True,
        "default_params": {"alpha": 1.0},
        "search_space": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    },

    # ── LAYER 1 — Low prediction ──────────────────────────────────────────────
    "xgboost_low": {
        "model_class": "xgboost.XGBRegressor",
        "target": "target_low",
        "requires_scaling": False,
        "default_params": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        },
        "search_space": {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        },
    },

    "lightgbm_low": {
        "model_class": "lightgbm.LGBMRegressor",
        "target": "target_low",
        "requires_scaling": False,
        "default_params": {
            "n_estimators": 400,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "search_space": {
            "n_estimators": [200, 400, 600],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    },

    "catboost_low": {
        "model_class": "catboost.CatBoostRegressor",
        "target": "target_low",
        "requires_scaling": False,
        "default_params": {
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
        },
        "search_space": {
            "iterations": [200, 400, 600],
            "depth": [4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    },

    "ridge_low": {
        "model_class": "sklearn.linear_model.Ridge",
        "target": "target_low",
        "requires_scaling": True,
        "default_params": {"alpha": 1.0},
        "search_space": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    },

    # ── LAYER 2 — Meta-learner (stacking) ────────────────────────────────────
    "meta_high": {
        "model_class": "sklearn.linear_model.Ridge",
        "target": "target_high",
        "requires_scaling": True,
        "default_params": {"alpha": 0.1, "fit_intercept": True},
        "search_space": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    },

    "meta_low": {
        "model_class": "sklearn.linear_model.Ridge",
        "target": "target_low",
        "requires_scaling": True,
        "default_params": {"alpha": 0.1, "fit_intercept": True},
        "search_space": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    },

    # ── Direction classifier ──────────────────────────────────────────────────
    "direction_classifier": {
        "model_class": "xgboost.XGBClassifier",
        "target": "target_direction",
        "requires_scaling": False,
        "default_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        },
        "search_space": {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    },
}
