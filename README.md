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
│   ├── run_train_dlinear.py
│   ├── run_train_xgboost.py
│   ├── run_backtest.py
│   └── run_predict.py
│
├── src/                  # Source code
│   ├── data/             #   Data pipeline
│   │   ├── loader.py             Parquet data loader
│   │   ├── preprocess.py         Factor preprocessing
│   │   └── dataset_builder.py    Sliding window dataset
│   ├── models/           #   Model implementations (6 models)
│   ├── train/            #   Training logic
│   ├── backtest/         #   Backtesting engine
│   │   ├── engine.py             Backtest main loop
│   │   ├── metrics.py            Performance metrics
│   │   └── portfolio.py          Portfolio construction
│   ├── predict/          #   Inference
│   └── utils/            #   Utilities (logging, seeding)
│
├── tests/                # Unit tests
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
