## Project Structure

```
JulongQuant/
│
├── dataset/              # Data storage
│   ├── input/            #   CSMAR raw data, teacher validation
│   ├── processed/        #   factor_panel_12.parquet, daily raw panels
│   └── output/           #   Experiment outputs & backtest results
│
├── reports/              # Generated experiment reports
│
├── scripts/              # Runnable entry points
│   ├── check_data_to_model.py    End-to-end smoke test
│   ├── run_experiment.py         Full experiment pipeline
│   ├── check_experiment.py       Output validation & baseline comparison
│   ├── build_factor_panel.py     CSMAR → unified daily panel
│   ├── calc_factors.py           12 factor + 2 targets computation
│   ├── select_stocks.py          Stock universe selection
│   └── ensemble.py               5-model ensemble & comparison
│
├── src/                  # Source code
│   ├── data/             #   Data pipeline
│   │   ├── loader.py             Parquet data loader
│   │   ├── preprocess.py         Factor preprocessing
│   │   └── dataset_builder.py    Sliding window dataset
│   ├── models/           #   Model implementations (6 models)
│   │   ├── lightgbm_model.py     LightGBM (tabular)
│   │   ├── xgboost_model.py      XGBoost (tabular)
│   │   ├── dlinear.py            DLinear (sequence)
│   │   ├── itransformer.py       iTransformer (sequence)
│   │   ├── patchtst.py           PatchTST (sequence)
│   │   └── tsmixer.py            TSMixer (sequence)
│   ├── train/            #   Training logic
│   │   ├── train_tabular.py      LightGBM & XGBoost training
│   │   └── train_torch.py        PyTorch sequence model training
│   ├── backtest/         #   Backtesting engine
│   │   ├── metrics.py            Sharpe, max drawdown, IC, turnover
│   │   ├── portfolio.py          Portfolio construction (top-N, equal)
│   │   └── engine.py             Date-by-date simulation
│   ├── predict/          #   Prediction
│   │   └── generate_predictions.py   Model inference & output
│   └── utils/            #   Utilities
│       ├── seed.py               Reproducibility
│       └── logger.py             Structured logging
│
├── tests/                # Unit tests (25 tests)
│   ├── test_data_alignment.py    X/y/meta alignment checks
│   ├── test_no_future_leakage.py Future-leakage prevention
│   └── test_portfolio_weight.py  Weight validity & constraints
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## License and Disclaimer

This project is released for non-commercial research and educational use only.

Commercial use is prohibited without prior written permission from the authors.

This project is not financial advice, investment advice, trading advice, or a recommendation to buy, sell, hold, or trade any financial instrument. Backtested or simulated performance does not guarantee future results. Use at your own risk.

See [LICENSE](./LICENSE) for details.
