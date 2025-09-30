# Crypto ETL Streamlit Project

## Overview
The Crypto ETL Streamlit project is designed to extract, transform, and load (ETL) cryptocurrency data from various sources. It provides a user-friendly interface using Streamlit, allowing users to visualize and interact with the processed data.

## Project Structure
```
crypto-etl-streamlit/
├── etl/
│   ├── extract.py       # Extracts data from APIs or databases
│   ├── transform.py     # Transforms and cleans the extracted data
│   ├── dq_checks.py     # Implements data quality checks
│   ├── load.py          # Loads transformed data into a target destination
│   └── pipeline.py      # Orchestrates the ETL process
├── ui/
│   └── app.py           # Main entry point for the Streamlit UI
├── configs/
│   └── config.yaml      # Configuration settings for the project
├── data/
│   ├── raw/             # Directory for raw data files
│   └── processed/       # Directory for processed data files
├── logs/                # Directory for log files
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd crypto-etl-streamlit
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure your environment variables in the `.env` file.

5. Run the Streamlit application:
   ```
   streamlit run ui/app.py
   ```

## Usage Guidelines
- Run `python -m etl.pipeline` to populate the SQLite warehouse with the latest market and price-history data.
- Launch the Streamlit workbench with `streamlit run ui/app.py`.
- Use the sidebar to:
  - Apply global filters (coins, market capitalization, performance, and history range).
  - Choose which widgets should appear on the dashboard and configure each widget individually.
  - Save personal layouts to `data/dashboards/` or reload previously stored presets.
- Supplementary datasets (for example `data/processed/sample_portfolio.csv`) can be registered in `configs/config.yaml` and surface additional widgets such as portfolio views.
- Monitor the logs in the `logs/` directory for any issues during the ETL process.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.