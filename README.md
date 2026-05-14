## Project Structure

```
JulongQuant/
│
├── dataset/              # Data storage
│   ├── input/            #   Raw input data (parquet files)
│   ├── processed/        #   Pre-processed factors
│   └── output/           #   Model predictions & backtest results
│
├── reports/              # Backtest reports and analysis
│
├── scripts/              # Runnable entry points
│   └── check_data_to_model.py    End-to-end smoke test
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
│   ├── backtest/         #   Backtesting engine (planned)
│   ├── predict/          #   Prediction
│   │   └── generate_predictions.py   Model inference & output
│   └── utils/            #   Utilities
│       ├── seed.py               Reproducibility
│       └── logger.py             Structured logging
│
├── tests/                # Unit tests (planned)
│   ├── test_data_alignment.py
│   ├── test_no_future_leakage.py
│   └── test_portfolio_weight.py
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
