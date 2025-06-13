# HR Attrition Dashboard

An interactive Dashboard built with Dash & Plotly to explore and analyze employee attrition drivers.  
This repository contains:
- A Jupyter notebook (`preprocessing.ipynb`) for cleaning and preparing the dataset.
- The original raw dataset from Kaggle (`data/raw/Cleaned_Employee_Data.xlsx`).
- The cleaned CSV (`data/processed/employee_attrition_clean.csv`) used by the Dash app.
- The Dash application script (`app.py`) which can be run directly to launch the dashboard.
- Custom styling in `assets/custom.css` for a dark‐mode sidebar and improved navigation menu.  


## Overview

This dashboard helps HR professionals, people ops managers and analysts explore patterns in employee attrition. It visualizes how demographics, job details and satisfaction scores relate to attrition, providing interactive filters and plots.

Key pages in the dashboard:

- **About**: Overview of the “Visualization Canvas” concept, story, data description, and tools used.
- **Data Explorer**: Compare two categorical fields via grouped bar charts (counts or percentages).
- **Satisfaction-driven Attrition**: Pick a grouping/filter variable (excluding satisfaction fields) to view Job, Environment, and Relationship satisfaction distributions by Attrition, faceted by group.
- **Attrition vs Retention**: Diverging bar chart showing Attrition vs Retention percentages across a selected categorical factor.
- **Attrition Reasons**: Predefined key analyses, including:
  - Total Working Years boxplot by Attrition
  - Attrition Rate by Age Group (line plot)
  - Attrition Rate heatmap: Job Level vs OverTime
  - Attrition Rate by Distance From Home bins (line plot)
  - Distribution of Monthly Income by Attrition (hist + KDE)
  - Distribution of Years With Current Manager by Attrition (hist + KDE)

## Setup & Run

1. **Clone the repository**  
   ```bash[
   git clone https://github.com/lianaaghamalyan/hr-attrition-dashboard.git
   ```
2. Run app.py
   ```bash[
   python app.py 
   ```
3. Visit http://127.0.0.1:8050/ in your browser to explore the HR Attrition Dashboard.
4. If you need to run on a different port, modify the app.run_server(...) call in app.py.
