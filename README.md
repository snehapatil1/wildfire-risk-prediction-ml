# Wildfire Risk Prediction System

## Overview

Wildfires pose a growing threat to ecosystems, public safety, and infrastructure across the United States. Fueled by climate change and prolonged droughts, their frequency and severity are on the rise. This project leverages machine learning and two decades of historical wildfire data to build a robust **Wildfire Risk Prediction System** that classifies fire events into **Low** or **High Risk** zones.

The goal is to enable data-driven decision-making that supports early warning systems, resource allocation, and long-term risk mitigation strategies.


## Dataset

I've used the **FPA-FOD (Fire Program Analysis - Fire Occurrence Dataset)**, which includes:
- 1.88 million geo-referenced wildfire records (1992–2015)
- Data across all U.S. states and federal lands
- Detailed attributes such as cause, size, location, and reporting agency

Key table used is **Fires**.
Since the data is too large to upload, it can be accessed directly on the kaggle link here : https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/data


## ML Pipeline

### 1. **Data Exploration & Preprocessing**
- Extracted relevant attributes
- Cleaned and merged relational SQLite tables
- Handled missing values, transformed date fields, and standardized categories

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed spatio-temporal fire trends
- Investigated fire size classes and cause-based patterns

### 3. **Feature Engineering**
- Converted `DISCOVERY_DOY` to month
- Simplified and grouped `STAT_CAUSE_DESCR`
- Mapped `FIRE_SIZE` to size classes (A–G)
- Created new categorical and geographical features

### 4. **Model Development**
- Models Trained: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, AdaBoost, k-NN
- Final Model: **Gradient Boost** with hyperparameter tuning via **RandomizedSearchCV**

### 5. **Binary Risk Classification**
- Target: Classify wildfires as either **High Risk** or **Low Risk**
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

### 6. **Application Integration**
- Developed a interactive minimal app interface using Streamlit
- User inputs: State, City, Fire Cause
- Output: Predicted Wildfire Risk


## Key Findings

- **Top Predictive Features**: `STAT_CAUSE_DESCR`, `LATITUDE`, `LONGITUDE`, `STATE`
- High-risk fires showed strong seasonal and geographic patterns
- Historical data can effectively inform real-time risk classification


## Conclusion

This project demonstrates how machine learning and historical wildfire data can be combined to:
- Proactively identify high-risk areas
- Improve disaster preparedness
- Support environmental conservation efforts


## Tech Stack

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, SQLite3
- **ML Tools**: RandomizedSearchCV, Classification Metrics
- **App Framework**: Streamlit
- **Data Source**: [FPA-FOD Dataset (Kaggle)](https://www.kaggle.com/datasets/)


## How to Run
1. Clone the repository:

```
git clone https://github.com/snehapatil1/wildfire-risk-prediction-ml
```

2. Install dependencies

```
pip3 install -r requirements.txt
```
3. Run Jupyter Notebook

```
jupyter notebook wildfire_app.ipynb
```
4. Run Streamlit app

```
streamlit run wildfire_app.py
```


## References and Citations
- Dataset Citation: Short, Karen C. 2017. Spatial wildfire occurrence data for the United States, 1992-2015 [FPA_FOD_20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
- Kaggle Reference: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/data
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Documentation: https://scikit-learn.org/
- Folium Documentation: https://python-visualization.github.io/folium/


## Notes
- ML code is in wildfire_app.ipynb.
- Streamlit app code is in wildfire_app.py, with configurations in config.py.
- Model and encoder artifacts are saved as .pkl for deployment.
- Project report is included in the repository (Report Wildfire Risk Prediction System.pdf).
- A video demo of the project is available on YouTube (https://youtu.be/Bs7Sz5Uom0M?si=R8Gn5Yv3CvQEG2HA).


> “An ounce of prevention is worth a pound of cure — and a gigabyte of data.”
