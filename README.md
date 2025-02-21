# Electricity Price Analysis & Forecasting in Europe ⚡

This project conducts an in-depth quantitative analysis of electricity prices across 31 European countries from January 2015 to December 2024. It leverages statistical and machine learning methods to understand historical trends, assess the impact of major disruptive events (such as the COVID-19 pandemic and the Russian invasion of Ukraine), and forecast future price movements.

📑 **Table of Contents**

**Project Overview**
- Data Description
- Data Cleaning
- Data Analysis & Insights
- Forecasting Models
- Results & Evaluation
- Technologies
- Getting Started
- Usage
- Authors

🔍 **Project Overview**

This project aims to:

- **Analyze** the evolution of electricity prices across Europe over a decade.
- **Identify** key trends and anomalies associated with major events (COVID-19, geopolitical conflicts).
- **Forecast** future prices using multiple predictive models.
- **Evaluate** model performance using standard metrics (MAE, MSE, MAPE).

📊 **Data Description**

The dataset comprises monthly electricity price observations (in EUR per MWhe) for 31 European countries covering the period from January 2015 to December 2024. Although the raw data represents European wholesale electricity prices, it has been refined to better reflect consumer-relevant trends.

🔗 **Data Sources**

**European Wholesale Electricity Prices(monthly):** Sourced from Ember Energy and other public datasets.
Additional sources (e.g., Eurostat, Kaggle) were reviewed, but only the dataset with sufficient observations was utilized.

🧹**Data Cleaning**

The raw data was meticulously cleaned and preprocessed to ensure robust analysis:

Filtering: Removed observations from countries with anomalous flat price evolutions (e.g., Montenegro, North Macedonia).

Imputation: For Bulgaria, Croatia, Serbia, and the United Kingdom, missing price values in early 2015 were replaced with the monthly average across the dataset.

Organization: Data was sorted by country and date to facilitate reliable time series analysis.

📈 **Data Analysis & Insights**

Key findings from the exploratory data analysis include:

**Stable Periods:** Electricity prices remained below 60 EUR until mid-2021.

**Disruptive Events:** A dramatic surge in prices was observed following the COVID-19 pandemic and the Ukraine crisis—with prices peaking around 400 EUR per MWhe 🚀.

**Stability Correlation:** Countries with a higher share of renewable energy (e.g., Sweden, Norway, Finland) exhibited more price stability during periods of market disruption 🌱.

Visualizations—such as time series plots, box plots, and variance analysis charts—were generated to illustrate these insights.

🤖 **Forecasting Models**

Three predictive modeling approaches were implemented to forecast future electricity prices:

**ARIMA (AutoRegressive Integrated Moving Average):**
Captures linear trends and seasonality 📉.
Challenges: Produced some anomalous forecasts (e.g., negative values) due to data non-linearity ⚠️.

**Hybrid Model (ARIMA + LSTM):**
Combines ARIMA for linear trends with LSTM (Long Short-Term Memory) to model residual (non-linear) patterns 🤝.
Result: Achieved the most promising and robust forecasts.

**Linear Regression:**
Provided straightforward linear forecasts 📈.
Outcome: Acceptable results; however, its performance was inferior compared to the hybrid approach.
Model performance was evaluated using MAE, MSE, and MAPE metrics.

🏆 **Results & Evaluation**

**Hybrid Model:** Outperformed ARIMA and Linear Regression by effectively combining the strengths of both models 🌟.

**Evaluation Metrics:** Although the hybrid model showed the best overall performance, the error metrics indicate room for further parameter tuning or alternative modeling techniques.

💻 **Technologies**

**Programming Language:** Python 

**Key Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow

**Additional Tools:** SAS Viya Data Analytics Platform (for advanced visualizations)

🚀 **Getting Started**

To get started with this project:

**Clone the Repository:**
git clone https://github.com/mervepakcan/Household-Electricity-Prices-in-Europe
cd Household-Electricity-Prices-in-Europe

**Install Dependencies:**
pip install -r requirements.txt

⚙️ **Usage**

You can reproduce the analysis by running the provided Jupyter Notebook or Python script:

**Jupyter Notebook:** project_python.ipynb
**Python Script:** py1.py
For example, to run the script:

python py1.py
The project workflow includes:

- Data ingestion and cleaning.
- Exploratory data analysis with visualizations.
- Implementation of forecasting models.
- Evaluation of model performance using MAE, MSE, and MAPE.


