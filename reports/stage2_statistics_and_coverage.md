# BEYOND SENTIMENT: DATASET STATISTICS & COVERAGE REPORT

**Generated:** 2026-11-03 01:37:02

## 1. Company-Wise Market & News Coverage

| Ticker | Entity Name | Sector | Trading Sessions | Direct News Count | Direct News Presence (%) | Mean Close (INR) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **RELIANCE** | Reliance Industries Limited | Energy | 1,541 | 5,017 | 81.44% | Rs 2,123.12 |
| **TCS** | Tata Consultancy Services Limited | Information Technology | 1,541 | 1,842 | 45.49% | Rs 3,293.41 |
| **INFY** | Infosys Limited | Information Technology | 1,541 | 1,906 | 48.35% | Rs 1,451.75 |
| **HDFCBANK** | HDFC Bank Limited | Financial Services | 1,541 | 2,737 | 63.92% | Rs 1,455.86 |
| **ICICIBANK** | ICICI Bank Limited | Financial Services | 1,541 | 1,367 | 43.54% | Rs 907.47 |
| **SBIN** | State Bank of India | Financial Services | 1,541 | 2,463 | 66.32% | Rs 577.72 |
| **HINDUNILVR** | Hindustan Unilever Limited | Fast Moving Consumer Goods | 1,541 | 1,023 | 30.95% | Rs 2,404.67 |
| **ITC** | ITC Limited | Fast Moving Consumer Goods | 1,541 | 1,547 | 42.96% | Rs 331.38 |
| **LT** | Larsen & Toubro Limited | Construction & Infrastructure | 1,541 | 1,221 | 42.57% | Rs 2,424.09 |
| **BHARTIARTL** | Bharti Airtel Limited | Telecommunications | 1,541 | 2,095 | 56.13% | Rs 1,032.66 |
| **SUNPHARMA** | Sun Pharmaceutical Industries Limited | Healthcare & Pharmaceuticals | 1,541 | 536 | 21.22% | Rs 1,107.08 |
| **CIPLA** | Cipla Limited | Healthcare & Pharmaceuticals | 1,541 | 536 | 18.95% | Rs 1,111.30 |
| **MARUTI** | Maruti Suzuki India Limited | Automobile | 1,541 | 1,073 | 37.25% | Rs 9,683.43 |
| **M&M** | Mahindra & Mahindra Limited | Automobile | 1,541 | 969 | 35.63% | Rs 1,660.01 |
| **TATASTEEL** | Tata Steel Limited | Metals & Mining | 1,541 | 1,056 | 36.73% | Rs 426.61 |

## 2. Sectoral Aggregates

| Sector | Total Records | Total Direct Articles | Direct News Presence |
| :--- | :---: | :---: | :---: |
| Automobile | 3,082 | 2,042 | 36.4% |
| Construction & Infrastructure | 1,541 | 1,221 | 42.6% |
| Energy | 1,541 | 5,017 | 81.4% |
| Fast Moving Consumer Goods | 3,082 | 2,570 | 37.0% |
| Financial Services | 4,623 | 6,567 | 57.9% |
| Healthcare & Pharmaceuticals | 3,082 | 1,072 | 20.1% |
| Information Technology | 3,082 | 3,748 | 46.9% |
| Metals & Mining | 1,541 | 1,056 | 36.7% |
| Telecommunications | 1,541 | 2,095 | 56.1% |


## 3. Storage and File Catalog

| Deliverable File | Disk Size | Record Count | Description |
| :--- | :---: | :---: | :--- |
| `market_raw_ohlcv.parquet` | ~1.4 MB | 23,130 | Raw nominal Bhavcopy OHLCV |
| `market_adjusted_ohlcv.parquet` | ~1.5 MB | 23,130 | Adjusted Total Return OHLCV |
| `corporate_actions.parquet` | ~35 KB | 150 | Official corporate action records |
| `news_clean.parquet` | ~30 MB | 155,537 | Extracted article-level text & metadata |
| `trading_calendar.parquet` | ~25 KB | 1,542 | Official NSE trading calendar & forward map |
| `final_integrated_dataset.parquet` | ~4.8 MB | 23,115 | Master analytical feature store |
