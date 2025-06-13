import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# ----------------------------
# 1. Load and preprocess data
# ----------------------------
DF_PATH = 'data/processed/employee_attrition_clean.csv'
try:
    df = pd.read_csv(DF_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {DF_PATH}. Please place it in the working directory.")

def add_derived_columns(df_in):
    """
    Add categorical buckets for Age, YearsAtCompany, and MonthlyIncome.
    """
    df2 = df_in.copy()
    if 'Age' in df2.columns:
        try:
            bins = [18, 23, 28, 33, 38, 43, 48, 53, 58, 60]
            labels = ['18-23', '23-28', '28-33', '33-38', '38-43', '43-48', '48-53', '53-58', '58-60']
            df2['AgeGroup'] = pd.cut(df2['Age'], bins=bins, labels=labels).astype('category')
        except Exception:
            pass
    if 'YearsAtCompany' in df2.columns:
        try:
            max_yr = int(df2['YearsAtCompany'].max())
            bins2 = [0, 3, 7, max_yr + 1]
            labels2 = ['0-3 yrs', '4-7 yrs', '8+ yrs']
            df2['YearsAtCompanyGroup'] = pd.cut(df2['YearsAtCompany'], bins=bins2, labels=labels2).astype('category')
        except Exception:
            pass
    if 'MonthlyIncome' in df2.columns:
        try:
            df2['IncomeGroup'] = pd.qcut(df2['MonthlyIncome'], q=4,
                                          labels=['Low', 'Mid-Low', 'Mid-High', 'High']).astype('category')
        except Exception:
            try:
                med = df2['MonthlyIncome'].median()
                df2['IncomeGroup'] = pd.cut(
                    df2['MonthlyIncome'],
                    bins=[df2['MonthlyIncome'].min() - 1, med, df2['MonthlyIncome'].max() + 1],
                    labels=['Below Median', 'Above Median']
                ).astype('category')
            except Exception:
                pass
    return df2

df = add_derived_columns(df)

# ----------------------------
# 2. Prepare column lists
# ----------------------------
categorical_columns = [
    c for c in df.columns
    if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object
]
numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
groupable_columns = categorical_columns + numeric_columns

# Satisfaction columns to exclude from grouping in the “Satisfaction-driven Attrition” page
SAT_COLS = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']
groupable_no_sat = [col for col in groupable_columns if col not in SAT_COLS]

# For Distribution Explorer: exclude satisfaction columns
explorer_columns = [c for c in categorical_columns if c not in set(SAT_COLS)]

# Ensure key columns exist
for sat in SAT_COLS:
    if sat not in df.columns:
        raise ValueError(f"Column '{sat}' not found in data.")
if 'Attrition' not in df.columns:
    raise ValueError("Column 'Attrition' not found in data.")

# ----------------------------
# 3. Initialize Dash app
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server

# ----------------------------
# 4. Sidebar
# ----------------------------
sidebar = html.Div(
    [
        html.H2("Dashboard", className="display-6", style={'padding-left': '10px'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("About", href="/", active="exact"),
                dbc.NavLink("Data Explorer", href="/distribution", active="exact"),
                dbc.NavLink("Satisfaction-driven Attrition", href="/groups", active="exact"),
                dbc.NavLink("Attrition vs Retention", href="/diverging", active="exact"),
                dbc.NavLink("Attrition Reasons", href="/reasons", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    },
)

# ----------------------------
# 5. Hidden placeholder for groupby-filter
# ----------------------------
# This invisible RangeSlider ensures Dash always registers id='groupby-filter',
# avoiding “nonexistent object” errors when callbacks reference it.
hidden_filter = html.Div(
    dcc.RangeSlider(id='groupby-filter', min=0, max=1, value=[0, 1]),
    style={'display': 'none'}
)

# ----------------------------
# 6. Page layouts
# ----------------------------
def page_about():
    return html.Div([
        html.H2("Visualization Canvas", className="mb-4"),
        html.P(
            "This dashboard is built around a “Visualization Canvas” idea: "
            "a space where HR analysts can choose variables and view charts dynamically. "
            "Rather than a static plot, we outline how to add dropdowns and a Graph component "
            "to select X/Y and color-by variables, generating scatter, box or bar charts on demand."
        ),
        html.H4("Story", className="mt-5"),
        html.P(
            "We examine factors behind employee attrition: demographics, job details and satisfaction—"
            "to uncover patterns that help HR improve retention."
        ),
        html.H4("Audience", className="mt-4"),
        html.P(
            "Designed for HR professionals, people ops managers, and decision-makers seeking data-driven insights "
            "into turnover. Also relevant for researchers and analysts in workforce studies."
        ),
        html.H4("Data", className="mt-4"),
        html.P("Kaggle HR Analytics Employee Attrition (~1,470 rows × ~35 columns)."),
        html.H4("Dataset Link", className="mt-4"),
        html.P([
            html.A(
                "Cleaned Employee Data on Kaggle",
                href="https://www.kaggle.com/datasets/anubhav761/hr-analytics-dashboard-employee-attrition/data?select=Cleaned_Employee_Data.xlsx",
                target="_blank"
            )
        ]),
        html.H4("Tools", className="mt-4"),
        html.Ul([
            html.Li("Jupyter Notebook for data loading, cleaning, analysis and feature engineering."),
            html.Li("Dash + Plotly for interactive charts and filters."),
            html.Li("Dash Bootstrap Components for consistent layout and styling."),
        ]),
        html.H2("How to Use", className="mt-5"),
        html.P("Use the left menu to navigate:"),
        html.Ul([
            html.Li([html.B("About"), ": overview and Visualization Canvas idea."]),
            html.Li([html.B("Data Explorer"), ": explore two categorical variables with grouped bars (counts or percentages)."]),
            html.Li([html.B("Satisfaction-driven Attrition"),
                     ": choose a grouping/filter variable—excluding satisfaction fields—to view satisfaction distributions (Job, Environment, Relationship) by Attrition, faceted by group."]),
            html.Li([html.B("Attrition vs Retention"),
                     ": diverging bar plot of Attrition vs Retention across any categorical factor."]),
            html.Li([html.B("Attrition Reasons"),
                     ": key analyses: Total Working Years boxplot; Attrition by AgeGroup; JobLevel vs OverTime heatmap; Attrition by DistanceFromHome bins; MonthlyIncome distribution; YearsWithCurrManager distribution."]),
        ]),
        html.Div(
            "Authors: Liana Aghamalyan & Zvart Aleksanyan",
            style={'textAlign': 'right', 'fontStyle': 'italic', 'marginTop': '3rem'}
        ),
    ], style={'margin-left': '18rem', 'padding': '2rem 1rem'})

def page_groups():
    return html.Div([
        html.H3("Attrition Groups & Satisfaction Distributions", className="mb-4"),
        html.Ul([
            html.Li("Select a grouping/filter variable (categorical or numeric), excluding satisfaction variables."),
            html.Li("Adjust the filter: choose level(s) or set a numeric range."),
            html.Li("View three plots: distributions of Job, Environment, and Relationship satisfaction by Attrition, faceted by group."),
        ], style={'margin-top': '0', 'margin-bottom': '1rem'}),
        dbc.Row([
            dbc.Col([
                html.Label("Group by:"),
                dcc.Dropdown(
                    id='groupby-var-dropdown',
                    options=[{'label': col, 'value': col} for col in groupable_no_sat],
                    value=(groupable_no_sat[0] if groupable_no_sat else None),
                    clearable=False,
                ),
            ], width=4),
            dbc.Col(html.Div(id='groupby-filter-container'), width=8),
        ], className="mb-4"),
        dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id='groupby-js-fig')), width=12), className="mb-4"),
        dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id='groupby-es-fig')), width=12), className="mb-4"),
        dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id='groupby-rs-fig')), width=12), className="mb-4"),
        html.Div(id='groupby-insight', style={'marginTop': '1rem', 'font-style': 'italic'}),
    ], style={'margin-left': '18rem', 'padding': '2rem 1rem'})

def page_reasons():
    # Compute six key plots; similar to original
    if 'TotalWorkingYears' in df.columns and 'Attrition' in df.columns:
        fig1 = px.box(df, x='Attrition', y='TotalWorkingYears',
                      title="Total Working Years by Attrition",
                      labels={'Attrition': 'Attrition', 'TotalWorkingYears': 'Total Working Years'})
    else:
        fig1 = px.scatter(title="Missing TotalWorkingYears or Attrition")

    # Attrition rate by AgeGroup
    if 'Attrition' in df.columns:
        if 'AgeGroup' in df.columns:
            age_ser = df['AgeGroup']
        elif 'Age' in df.columns:
            try:
                bins = [df['Age'].min() - 1] + list(np.linspace(df['Age'].min(), df['Age'].max(), 6)) + [df['Age'].max() + 1]
                bins = sorted(set(bins))
                labels = [f"[{int(np.floor(bins[i] + 1))}, {int(np.floor(bins[i+1]))}]" for i in range(len(bins)-1)]
                df['_tmpAgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
                age_ser = df['_tmpAgeGroup']
            except Exception:
                age_ser = pd.Series(['Unknown'] * len(df), index=df.index)
        else:
            age_ser = pd.Series(['Unknown'] * len(df), index=df.index)

        temp = pd.DataFrame({'AgeGroup': age_ser.astype(str), 'Attrition': df['Attrition'].astype(str)}).dropna(subset=['AgeGroup', 'Attrition'])
        if not temp.empty:
            temp['AttrNum'] = temp['Attrition'].map({'Yes': 1, 'No': 0})
            grp = temp.groupby('AgeGroup')['AttrNum'].mean().reset_index()
            try:
                if hasattr(age_ser, "cat") and pd.api.types.is_categorical_dtype(age_ser):
                    grp['AgeGroup'] = pd.Categorical(grp['AgeGroup'], categories=age_ser.cat.categories, ordered=True)
                    grp = grp.sort_values('AgeGroup')
                else:
                    grp = grp.sort_values('AttrNum', ascending=False)
            except Exception:
                pass
            fig2 = px.line(grp, x='AgeGroup', y='AttrNum', markers=True,
                           title="Attrition Rate by Age Group",
                           labels={'AgeGroup': 'Age Group', 'AttrNum': 'Attrition Rate'})
            fig2.update_layout(yaxis_tickformat='.0%', xaxis_title="Age Group")
        else:
            fig2 = px.scatter(title="No data for AgeGroup / Attrition rate")
        if '_tmpAgeGroup' in df.columns:
            df.drop(columns=['_tmpAgeGroup'], inplace=True)
    else:
        fig2 = px.scatter(title="Missing Attrition column")

    if all(col in df.columns for col in ['JobLevel', 'OverTime', 'Attrition']):
        temp3 = df[['JobLevel', 'OverTime', 'Attrition']].dropna().copy()
        temp3['AttrNum'] = temp3['Attrition'].map({'Yes': 1, 'No': 0})
        pivot = temp3.groupby(['JobLevel', 'OverTime'])['AttrNum'].mean().unstack(fill_value=np.nan)
        z = pivot.values
        x_labels = list(pivot.columns.astype(str))
        y_labels = list(pivot.index.astype(str))
        fig3 = px.imshow(z, x=x_labels, y=y_labels, color_continuous_scale='Blues',
                         title="Attrition Rate by Job Level and OverTime",
                         labels=dict(x="OverTime", y="JobLevel", color="Attrition Rate"))
        fig3.update_traces(
            text=np.round(pivot.values, 2),
            texttemplate="%{text:.2f}", textfont={"size": 12},
            hovertemplate="JobLevel=%{y}<br>OverTime=%{x}<br>Attrition Rate=%{z:.2f}<extra></extra>"
        )
        fig3.update_coloraxes(colorbar_tickformat=".0%")
    else:
        fig3 = px.scatter(title="Missing JobLevel, OverTime, or Attrition")

    if 'DistanceFromHome' in df.columns and 'Attrition' in df.columns:
        df_local = df.copy()
        df_local['AttrNum'] = df_local['Attrition'].map({'Yes': 1, 'No': 0})
        df_local = df_local.dropna(subset=['AttrNum', 'DistanceFromHome'])
        if df_local.empty:
            fig4 = px.scatter(title="No valid data for DistanceFromHome / Attrition")
        else:
            try:
                dist_bins = pd.qcut(df_local['DistanceFromHome'], q=4, duplicates='drop')
                df_local['DistBin'] = dist_bins
            except Exception:
                try:
                    df_local['DistBin'] = pd.cut(df_local['DistanceFromHome'], bins=4)
                except Exception:
                    df_local['DistBin'] = "All"
            grp = df_local.groupby('DistBin')['AttrNum'].mean().reset_index(name='AttritionRate')
            if grp.empty:
                fig4 = px.scatter(title="No groups after binning DistanceFromHome")
            else:
                if pd.api.types.is_categorical_dtype(df_local['DistBin']):
                    cat_order = list(df_local['DistBin'].cat.categories)
                    grp['BinLabel'] = grp['DistBin'].astype(str)
                    ordered_labels = [str(c) for c in cat_order]
                    grp['BinLabel'] = pd.Categorical(grp['BinLabel'], categories=ordered_labels, ordered=True)
                    grp = grp.sort_values('BinLabel')
                else:
                    grp['BinLabel'] = grp['DistBin'].astype(str)
                    ordered_labels = None
                if ordered_labels:
                    fig4 = px.line(grp, x='BinLabel', y='AttritionRate', markers=True,
                                   title="Attrition Rate by Distance From Home",
                                   labels={'BinLabel': 'DistanceFromHome bin', 'AttritionRate': 'Attrition Rate'},
                                   category_orders={'BinLabel': ordered_labels})
                else:
                    fig4 = px.line(grp, x='BinLabel', y='AttritionRate', markers=True,
                                   title="Attrition Rate by Distance From Home",
                                   labels={'BinLabel': 'DistanceFromHome bin', 'AttritionRate': 'Attrition Rate'})
                fig4.update_layout(yaxis_tickformat='.0%')
                fig4.update_xaxes(tickangle=-45)
    else:
        fig4 = px.scatter(title="Missing DistanceFromHome or Attrition")

    if 'MonthlyIncome' in df.columns and 'Attrition' in df.columns:
        fig5 = go.Figure()
        for at_val, color in zip(['Yes', 'No'], ['orange', 'blue']):
            grp_vals = df[df['Attrition'] == at_val]['MonthlyIncome'].dropna().values
            if len(grp_vals) > 1:
                hist = np.histogram(grp_vals, bins=30, density=True)
                bin_edges, heights = hist[1], hist[0]
                fig5.add_trace(go.Bar(
                    x=bin_edges[:-1], y=heights, name=f"Hist {at_val}",
                    marker=dict(color=color), opacity=0.4,
                    width=(bin_edges[1] - bin_edges[0])
                ))
                std, n = np.std(grp_vals), len(grp_vals)
                if std > 0 and n > 1:
                    bw = 1.06 * std * (n ** (-1/5))
                    x_min, x_max = grp_vals.min(), grp_vals.max()
                    x_grid = np.linspace(x_min, x_max, 200)
                    if bw <= 0:
                        bw = (x_max - x_min) / 10.0
                    kde_vals = np.zeros_like(x_grid)
                    coeff = 1.0 / (np.sqrt(2 * np.pi) * bw * n)
                    for val in grp_vals:
                        kde_vals += np.exp(-0.5 * ((x_grid - val) / bw) ** 2)
                    kde_vals *= coeff
                    fig5.add_trace(go.Scatter(x=x_grid, y=kde_vals, mode='lines', name=f"KDE {at_val}", line=dict(color=color)))
        fig5.update_layout(
            title="Distribution of MonthlyIncome by Attrition",
            xaxis_title="MonthlyIncome",
            yaxis_title="Density",
            barmode='overlay'
        )
    else:
        fig5 = px.scatter(title="Missing MonthlyIncome or Attrition")

    if 'YearsWithCurrManager' in df.columns and 'Attrition' in df.columns:
        fig6 = go.Figure()
        for at_val, color in zip(['Yes', 'No'], ['orange', 'blue']):
            grp_vals = df[df['Attrition'] == at_val]['YearsWithCurrManager'].dropna().values
            if len(grp_vals) > 1:
                hist = np.histogram(grp_vals, bins=30, density=True)
                bin_edges, heights = hist[1], hist[0]
                fig6.add_trace(go.Bar(
                    x=bin_edges[:-1], y=heights, name=f"Hist {at_val}",
                    marker=dict(color=color), opacity=0.4,
                    width=(bin_edges[1] - bin_edges[0])
                ))
                std, n = np.std(grp_vals), len(grp_vals)
                if std > 0 and n > 1:
                    bw = 1.06 * std * (n ** (-1/5))
                    x_min, x_max = grp_vals.min(), grp_vals.max()
                    x_grid = np.linspace(x_min, x_max, 200)
                    if bw <= 0:
                        bw = (x_max - x_min) / 10.0
                    kde_vals = np.zeros_like(x_grid)
                    coeff = 1.0 / (np.sqrt(2 * np.pi) * bw * n)
                    for val in grp_vals:
                        kde_vals += np.exp(-0.5 * ((x_grid - val) / bw) ** 2)
                    kde_vals *= coeff
                    fig6.add_trace(go.Scatter(x=x_grid, y=kde_vals, mode='lines', name=f"KDE {at_val}", line=dict(color=color)))
        fig6.update_layout(
            title="Distribution of YearsWithCurrManager by Attrition",
            xaxis_title="YearsWithCurrManager",
            yaxis_title="Density",
            barmode='overlay'
        )
    else:
        fig6 = px.scatter(title="Missing YearsWithCurrManager or Attrition")

    return html.Div([
        html.H3("Attrition Reasons & Key Patterns", className="mb-4"),
        html.P("Below are selected plots derived from correlation and EDA:"),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig1, style={'height': '600px'})), width=10),
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig2, style={'height': '600px'})), width=10),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig3, style={'height': '600px'})), width=10),
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig4, style={'height': '600px'})), width=10),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig5, style={'height': '600px'})), width=10),
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig6, style={'height': '600px'})), width=10),
        ], className="mb-4"),
    ], style={'margin-left': '18rem', 'padding': '2rem 1rem'})

def page_distribution():
    return html.Div([
        html.H3("Distribution Explorer", className="mb-4"),
        html.P(
            "Choose a primary category and a comparison category to see how their distributions relate. "
            "Toggle between viewing raw counts or relative percentages."
        ),
        dbc.Row([
            dbc.Col([
                html.Label("Primary category:"),
                dcc.Dropdown(
                    id='dist-x-dropdown',
                    options=[{'label': col, 'value': col} for col in explorer_columns],
                    value=(explorer_columns[0] if explorer_columns else None),
                    clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("Comparison category:"),
                dcc.Dropdown(
                    id='dist-color-dropdown',
                    options=[{'label': col, 'value': col} for col in explorer_columns],
                    value=(explorer_columns[1] if len(explorer_columns)>1 else (explorer_columns[0] if explorer_columns else None)),
                    clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("Display as:"),
                dcc.RadioItems(
                    id='dist-mode-radio',
                    options=[
                        {'label': 'Count', 'value': 'count'},
                        {'label': 'Percent', 'value': 'percent'}
                    ],
                    value='count',
                    inline=True
                )
            ], width=4),
        ], className="mb-4"),
        dcc.Loading(dcc.Graph(id='dist-graph'), type='default'),
        html.Div(id='dist-insight', style={'marginTop':'1rem', 'font-style':'italic'}),
    ], style={'margin-left': '18rem', 'padding': '2rem 1rem'})

def page_diverging():
    available = [col for col in categorical_columns if col != 'Attrition']
    default_val = available[0] if available else None
    return html.Div([
        html.H3("Diverging Plot: Attrition vs Retention", className="mb-4"),
        html.P("Select a categorical factor to view Attrition vs Retention in a diverging bar chart."),
        dbc.Row([
            dbc.Col([
                html.Label("Factor:"),
                dcc.Dropdown(
                    id='diverge-factor-dropdown',
                    options=[{'label': col, 'value': col} for col in available],
                    value=default_val,
                    clearable=False
                )
            ], width=4),
        ], className="mb-4"),
        dcc.Loading(dcc.Graph(id='diverge-graph'), type='default'),
        html.Div(id='diverge-insight', style={'marginTop': '1rem', 'font-style': 'italic'}),
    ], style={'margin-left': '18rem', 'padding': '2rem 1rem'})

# ----------------------------
# 7. app.validation_layout and app.layout
# ----------------------------
app.validation_layout = html.Div([
    sidebar,
    page_about(),
    page_groups(),
    page_distribution(),
    page_diverging(),
    page_reasons(),
    hidden_filter,
])

hidden_filter = html.Div(
    dcc.RangeSlider(id='groupby-filter', min=0, max=1, value=[0,1]),
    style={'display': 'none'}
)

app.layout = html.Div([
    dcc.Location(id='url'),
    sidebar,
    html.Div(id='page-content'),
    hidden_filter,
])

# ----------------------------
# 8. Routing callback
# ----------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    if pathname in ["/", "/about", ""]:
        return page_about()
    elif pathname == "/groups":
        return page_groups()
    elif pathname == "/reasons":
        return page_reasons()
    elif pathname == "/distribution":
        return page_distribution()
    elif pathname == "/diverging":
        return page_diverging()
    else:
        return html.Div([
            html.H3("404: Page not found", className="mb-4"),
            html.P(f"The pathname {pathname} was not recognized."),
        ], style={'margin-left': '18rem', 'padding': '2rem 1rem'})

# ----------------------------
# 9. Callback: dynamic filter control for grouping
# ----------------------------
@app.callback(
    Output('groupby-filter-container', 'children'),
    Input('groupby-var-dropdown', 'value')
)
def render_groupby_filter(selected_var):
    if selected_var is None or selected_var not in df.columns:
        return html.Div("Select a valid variable.")
    # Numeric variable: show RangeSlider
    if pd.api.types.is_numeric_dtype(df[selected_var]):
        col_min = float(df[selected_var].min())
        col_max = float(df[selected_var].max())
        try:
            step = (col_max - col_min) / 5
            marks = {round(col_min + i * step, 2): str(int(round(col_min + i * step))) for i in range(6)}
        except:
            marks = None
        return html.Div([
            html.Label(f"Filter {selected_var} range:"),
            dcc.RangeSlider(
                id='groupby-filter',
                min=col_min,
                max=col_max,
                value=[col_min, col_max],
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False
            ),
        ])
    # Categorical variable: show multi-dropdown
    else:
        if pd.api.types.is_categorical_dtype(df[selected_var]):
            options = [{'label': str(cat), 'value': cat} for cat in df[selected_var].cat.categories]
        else:
            unique_vals = sorted(df[selected_var].dropna().unique().tolist())
            options = [{'label': str(val), 'value': val} for val in unique_vals]
        default = [opt['value'] for opt in options]
        return html.Div([
            html.Label(f"Select {selected_var} level(s):"),
            dcc.Dropdown(
                id='groupby-filter',
                options=options,
                value=default,
                multi=True,
                clearable=False,
            )
        ])

# ----------------------------
# 10. Callback: update satisfaction distribution plots
# ----------------------------
@app.callback(
    Output('groupby-js-fig', 'figure'),
    Output('groupby-es-fig', 'figure'),
    Output('groupby-rs-fig', 'figure'),
    Output('groupby-insight', 'children'),
    Input('groupby-var-dropdown', 'value'),
    Input('groupby-filter', 'value')
)
def update_groupby_plots(group_var, filter_value):
    dff = df.copy()
    if group_var is None or group_var not in df.columns:
        blank = px.scatter(title="Select valid grouping variable")
        return blank, blank, blank, ""
    # Apply filter if provided
    if filter_value is not None:
        if pd.api.types.is_numeric_dtype(df[group_var]):
            if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                low, high = filter_value
                try:
                    dff = dff[dff[group_var].between(low, high)]
                except:
                    pass
        else:
            if isinstance(filter_value, list):
                dff = dff[dff[group_var].isin(filter_value)]
    if dff.empty:
        blank = px.scatter(title="No data to display")
        return blank, blank, blank, ""

    # Determine grouping column: bin numeric or categorical directly
    if pd.api.types.is_numeric_dtype(df[group_var]):
        try:
            dff['__bin'] = pd.qcut(dff[group_var], q=4, duplicates='drop')
            dff['__bin_str'] = dff['__bin'].astype(str)
            group_col = '__bin_str'
        except Exception:
            dff['__bin_str'] = "All"
            group_col = '__bin_str'
    else:
        group_col = group_var

    def make_sat_plot(sat_col):
        if sat_col not in dff.columns or dff.empty:
            return px.bar(title=f"No data for {sat_col}")
        dff2 = dff.dropna(subset=[group_var, sat_col, 'Attrition']).copy()
        if dff2.empty:
            return px.bar(title=f"No data for {sat_col}")

        if pd.api.types.is_numeric_dtype(dff2[group_var]):
            try:
                dff2['__bin'] = pd.qcut(dff2[group_var], q=4, duplicates='drop')
                dff2['__group_str'] = dff2['__bin'].astype(str)
            except:
                dff2['__group_str'] = "All"
            grp_col = '__group_str'
        else:
            dff2['__group_str'] = dff2[group_var].astype(str)
            grp_col = '__group_str'

        grp_df = dff2.groupby([grp_col, 'Attrition', sat_col]).size().reset_index(name='Count')
        grp_df['Total'] = grp_df.groupby([grp_col, 'Attrition'])['Count'].transform('sum')
        grp_df['Pct'] = grp_df['Count'] / grp_df['Total'] * 100

        if pd.api.types.is_categorical_dtype(df[sat_col]):
            sat_order = list(df[sat_col].cat.categories)
        else:
            sat_order = sorted(dff2[sat_col].dropna().unique().tolist())

        levels = sorted(dff2[grp_col].unique())
        fig = px.bar(
            grp_df,
            x=grp_col,
            y='Pct',
            color=sat_col,
            facet_col='Attrition',
            barmode='stack',
            category_orders={sat_col: sat_order, grp_col: levels},
            labels={'Pct': 'Percentage (%)', grp_col: group_var},
            title=f"{sat_col} distribution by Attrition across {group_var}"
        )
        fig.update_layout(yaxis_tickformat='.0f', margin={'t': 50, 'b': 50}, bargap=0.15)
        return fig

    fig_js = make_sat_plot('JobSatisfaction')
    fig_es = make_sat_plot('EnvironmentSatisfaction')
    fig_rs = make_sat_plot('RelationshipSatisfaction')

    # Clean up temporary columns
    if pd.api.types.is_numeric_dtype(df[group_var]):
        for col in ['__bin', '__bin_str']:
            if col in dff.columns:
                dff.drop(columns=[col], inplace=True)

    return fig_js, fig_es, fig_rs, ""

# ----------------------------
# 11. Callback: Attrition Reasons by factor
# ----------------------------
@app.callback(
    Output('reason-graph', 'figure'),
    Output('reason-insight', 'children'),
    Input('reason-factor-dropdown', 'value')
)
def update_reason(factor):
    if factor is None or factor not in df.columns:
        blank = px.scatter(title="Select a valid variable")
        return blank, "Please select a valid factor."
    dff = df.dropna(subset=[factor, 'Attrition']).copy()
    if dff.empty:
        blank = px.scatter(title="No data to display")
        return blank, "No data available."
    grp = dff.groupby(factor)['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index(name='AttritionRate')
    if pd.api.types.is_categorical_dtype(df[factor]):
        grp[factor] = pd.Categorical(grp[factor], categories=df[factor].cat.categories, ordered=True)
        grp = grp.sort_values(factor)
        low, high = grp.iloc[0], grp.iloc[-1]
        insight = (
            f"Attrition at lowest {factor} = {low[factor]}: {low['AttritionRate']:.1%}, "
            f"highest {factor} = {high[factor]}: {high['AttritionRate']:.1%}."
        )
    else:
        grp = grp.sort_values('AttritionRate', ascending=False)
        top, bottom = grp.iloc[0], grp.iloc[-1]
        insight = (
            f"Highest attrition: {factor} = {top[factor]} at {top['AttritionRate']:.1%}. "
            f"Lowest attrition: {factor} = {bottom[factor]} at {bottom['AttritionRate']:.1%}."
        )
    fig = px.bar(grp, x=factor, y='AttritionRate', title=f"Attrition Rate by {factor}",
                 labels={'AttritionRate': 'Attrition Rate'})
    fig.update_layout(yaxis_tickformat='.0%')
    return fig, insight

# ----------------------------
# 12. Callback: Distribution Explorer
# ----------------------------
@app.callback(
    Output('dist-graph', 'figure'),
    Output('dist-insight', 'children'),
    Input('dist-x-dropdown', 'value'),
    Input('dist-color-dropdown', 'value'),
    Input('dist-mode-radio', 'value'),
)
def update_distribution(x_var, color_var, display_mode):
    if x_var is None or color_var is None or x_var not in df.columns or color_var not in df.columns:
        fig = px.scatter(title="Select valid variables")
        return fig, "Please select valid variables."
    dff = df.dropna(subset=[x_var, color_var]).copy()
    if dff.empty:
        fig = px.scatter(title="No data")
        return fig, "No data available."
    dff[x_var] = dff[x_var].astype(str)
    dff[color_var] = dff[color_var].astype(str)
    grp = dff.groupby([x_var, color_var]).size().reset_index(name='Count')
    if grp.empty:
        fig = px.scatter(title="No data")
        return fig, "No data available."
    if display_mode == 'percent':
        totals = grp.groupby(x_var)['Count'].transform('sum')
        grp['Percent'] = grp['Count'] / totals * 100
        fig = px.bar(
            grp,
            x=x_var,
            y='Percent',
            color=color_var,
            title=f"{color_var} distribution by {x_var} (Percent)",
            labels={'Percent': 'Percentage (%)'}
        )
        fig.update_layout(barmode='stack', yaxis_tickformat='.1f',
                          xaxis_title=x_var, legend_title=color_var,
                          margin={'t':50, 'b':50})
    else:
        fig = px.bar(
            grp,
            x=x_var,
            y='Count',
            color=color_var,
            barmode='group',
            title=f"{color_var} distribution by {x_var} (Count)",
            labels={'Count': 'Count'}
        )
        fig.update_layout(xaxis_title=x_var, legend_title=color_var,
                          margin={'t':50, 'b':50})
    insight_lines = []
    for lvl in sorted(dff[x_var].unique()):
        sub = grp[grp[x_var] == lvl]
        if sub.empty:
            continue
        if display_mode == 'percent':
            top = sub.loc[sub['Percent'].idxmax()]
            insight_lines.append(f"{lvl}: {top[color_var]} ({top['Percent']:.1f}%)")
        else:
            top = sub.loc[sub['Count'].idxmax()]
            insight_lines.append(f"{lvl}: {top[color_var]}")
    insight = html.Ul([html.Li(line) for line in insight_lines]) if insight_lines else ""
    return fig, insight

# ----------------------------
# 13. Callback: Diverging plot
# ----------------------------
@app.callback(
    Output('diverge-graph', 'figure'),
    Output('diverge-insight', 'children'),
    Input('diverge-factor-dropdown', 'value')
)
def update_diverging(factor):
    if factor is None or factor not in df.columns:
        blank = px.scatter(title="Select a valid variable")
        return blank, "Please select a valid categorical variable."
    dff = df.dropna(subset=[factor, 'Attrition']).copy()
    if dff.empty:
        blank = px.scatter(title="No data to display")
        return blank, "No data available."
    grp = dff.groupby([factor, 'Attrition'], as_index=False).agg(Count=('Attrition', 'size'))
    if grp.empty:
        blank = px.scatter(title="No data to display")
        return blank, "No data available after grouping."
    grp['Total'] = grp.groupby(factor)['Count'].transform('sum')
    grp['Pct'] = grp['Count'] / grp['Total'] * 100
    pivot = grp.pivot(index=factor, columns='Attrition', values='Pct').fillna(0)
    if 'Yes' not in pivot.columns:
        pivot['Yes'] = 0.0
    if 'No' not in pivot.columns:
        pivot['No'] = 0.0
    if pd.api.types.is_categorical_dtype(df[factor]):
        pivot = pivot.reindex(df[factor].cat.categories, fill_value=0)
    else:
        pivot = pivot.sort_index()
    pivot = pivot.copy()
    pivot['Yes_neg'] = -pivot['Yes']
    pivot['No_pos'] = pivot['No']
    plot_df = pivot.reset_index().rename(columns={pivot.index.name or factor: 'Level'})
    fig = px.bar(
        plot_df,
        y='Level',
        x=['Yes_neg', 'No_pos'],
        orientation='h',
        title=f"Diverging Attrition vs Retention by {factor}",
        labels={'value': 'Percentage', 'variable': 'Category'}
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
    for trace in fig.data:
        if trace.name == 'Yes_neg':
            trace.name = 'Attrition (Yes)'
            trace.hovertemplate = "Attrition (Yes): %{x:.1f}%<extra></extra>"
        elif trace.name == 'No_pos':
            trace.name = 'Retention (No)'
            trace.hovertemplate = "Retention (No): %{x:.1f}%<extra></extra>"
    fig.update_layout(margin={'t': 50, 'l': 150})
    try:
        highest_level = pivot['Yes'].idxmax()
        highest_val = pivot.loc[highest_level, 'Yes']
        insight = f"Highest attrition percentage: '{highest_level}' at {highest_val:.1f}%."
    except Exception:
        insight = ""
    return fig, insight

# ----------------------------
# 14. Run the app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=False)