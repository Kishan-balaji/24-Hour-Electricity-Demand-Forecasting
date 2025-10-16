
#  Smart Meter Energy Forecasting (Fast Track)

A **24-hour electricity demand forecasting pipeline** built using smart meter data.  
Predicts hourly energy usage for the next day (`T+1 … T+24`) using **Ridge Regression** and a **Seasonal-Naive baseline**.  
Includes automated **data preparation**, **light backtesting**, **visualizations**, and **PDF report generation**.

---


## Instructions to Download Dataset
Create a directory called "data" and download/place the dataset files under this directory
Download from https://www.kaggle.com/datasets/jehanbhathena/smart-meter-data-mathura-and-bareilly

---

## Folder Structure



```plaintext
data/
|-- CEEW - Smart meter data <City> <Year>.csv

results/
├── artifacts/
│   └── fast_track/
│       ├── metrics.csv
│       ├── forecast_T_plus_24.csv
│       └── plots/
│           ├── actual_vs_forecast.png
│           └── horizon_mae.png
└── reports/
    └── fast_track_report{city}{year}.pdf
```
---

## Input Data Format

Place your dataset in the `data/` folder as:  
`CEEW - Smart meter data <City> <Year>.csv`

| x_Timestamp | t_kWh | z_Avg Voltage (Volt) | z_Avg Current (Amp) | y_Freq (Hz) | meter |
|--------------|--------|----------------------|---------------------|-------------|--------|
| 2020-01-01 00:00:00 | 0.002 | 251.26 | 0.15 | 49.97 | BR02 |
| 2020-01-01 00:03:00 | 0.001 | 251.23 | 0.15 | 49.94 | BR02 |

---

## Dependency Installation

pip install -r requirements.txt

## Manual Dependency Installation
pip install pandas numpy matplotlib scikit-learn reportlab requests


## Run Code

Sample: 
python run_forecast.py --city Bareilly --year 2020 --with_weather --make_plots --save_report --history_window 7



| Argument                  | Description                             |
| ------------------------- | --------------------------------------- |
| `--city`                  | City name (must match dataset name)     |
| `--year`                  | Dataset year                            |
| `--with_weather`          | Include Open-Meteo temperature forecast |
| `--make_plots`            | Generate and save forecast plots        |
| `--save_report`           | Create PDF summary             |
| `--history_window`        | Use N days of history (default = 7)     |



## Outputs
| File                     | Description                                   |
| ------------------------ | --------------------------------------------- |
| `metrics.csv`            | Validation metrics for Ridge vs. Naive models |
| `forecast_T_plus_24.csv` | 24-hour forecast (p10, p50, p90 quantiles)    |
| `backtest_metrics.csv`   | Backtest metrics for previous 2 days          |
| `plots/`                 | Visualization outputs                         |
| `fast_track_report.pdf`  | Final report (summary + plots)         |
