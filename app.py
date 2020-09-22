import base64
import io
import textwrap

import dash
import dash_core_components as dcc
import dash_html_components as html
import gunicorn
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import flask
import pandas as pd
import urllib.parse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import math
import scipy.stats
import dash_table
from dash_table.Format import Format, Scheme
from colour import Color
import dash_bootstrap_components as dbc

# from waitress import serve

external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        "https://codepen.io/sutharson/pen/dyYzEGZ.css",
                        "https://fonts.googleapis.com/css2?family=Raleway&display=swap"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# "external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"
# https://raw.githubusercontent.com/aaml-analytics/pca-explorer/master/LoadingStatusStyleSheet.css
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
tabs_styles = {'height': '40px', 'font-family': 'Raleway', 'fontSize': 14}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'Weight': 'bold'
}

tab_selected_style = {
    'borderTop': '3px solid #333333',
    'borderBottom': '1px solid #d6d6d6 ',
    'backgroundColor': '#f6f6f6',
    'color': '#333333',
    # 'fontColor': '#004a4a',
    'fontWeight': 'bold',
    'padding': '6px'
}

# APP ABOUT DESCRIPTION
MOF_tool_about = textwrap.wrap(' These tools aim to provide a reproducible and consistent data visualisation platform '
                               'where experimental and computational researchers can use big data and statistical '
                               'analysis to find the best materials for specific applications. Principal Component '
                               'Analysis (PCA) is a dimension reduction technique that can be used to reduce a large '
                               'set of observable variables to a smaller set of latent variables that still contain '
                               'most of the information in the large set (feature extraction). This is done by '
                               'transforming a number of (possibly) correlated variables into some number of orthogonal '
                               '(uncorrelated) variables called principal components to find the directions of maximal '
                               'variance. PCA can be used to ease data visualisation by having fewer dimensions to plot '
                               'or be used as a pre-processing step before using another Machine Learning (ML)'
                               ' algorithm for regression '
                               'and classification tasks. PCA can be used to improve an ML algorithm performance, '
                               'reduce overfitting and reduce noise in data.',
                               width=50)
Scree_plot_about = textwrap.wrap(' The Principal Component Analysis Visualisation Tools runs PCA for the user and '
                                 'populates a Scree plot. This plot allows the user to determine if PCA is suitable '
                                 'for '
                                 'their dataset and if can compromise an X% drop in explained variance to '
                                 'have fewer dimensions.', width=50)
Feature_correlation_filter = textwrap.wrap("Feature correlation heatmaps provide users with feature analysis and "
                                           "feature principal component analysis. This tool will allow users to see the"
                                           " correlation between variables and the"
                                           " covariances/correlations between original variables and the "
                                           "principal components (loadings)."
                                           , width=50)
plots_analysis = textwrap.wrap('Users can keep all variables as features or drop certain variables to produce a '
                               'Biplot, cos2 plot and contribution plot. The score plot is used to look for clusters, '
                               'trends, and outliers in the first two principal components. The loading plot is used to'
                               ' visually interpret the first two principal components. The biplot overlays the score '
                               'plot and the loading plot on the same graph. The squared cosine (cos2) plot shows '
                               'the importance of a component for a given observation i.e. measures '
                               'how much a variable is represented in a component. The contribution plot contains the '
                               'contributions (%) of the variables to the principal components', width=50, )
data_table_download = textwrap.wrap("The user's inputs from the 'Plots' tab will provide the output of the data tables."
                                    " The user can download the scores, eigenvalues, explained variance, "
                                    "cumulative explained variance, loadings, "
                                    "cos2 and contributions from the populated data tables. "
                                    "Note: Wait for user inputs to be"
                                    " computed (faded tab app will return to the original colour) before downloading the"
                                    " data tables. ", width=50)
MOF_GH = textwrap.wrap(" to explore AAML's sample data and read more on"
                       " AAML's Principal Component Analysis Visualisation Tool Manual, FAQ's & Troubleshooting"
                       " on GitHub... ", width=50)

####################
# APP LAYOUT #
####################
fig = go.Figure()
fig1 = go.Figure()
app.layout = html.Div([
    html.Div([
        html.Img(
            src='https://raw.githubusercontent.com/aaml-analytics/mof-explorer/master/UOC.png',
            height='35', width='140', style={'display': 'inline-block', 'padding-left': '1%'}),
        html.Img(src='https://raw.githubusercontent.com/aaml-analytics/mof-explorer/master/A2ML-logo.png',
                 height='50', width='125', style={'float': 'right', 'display': 'inline-block', 'padding-right': '2%'}),
        html.H1("Principal Component Analysis Visualisation Tools",
                style={'display': 'inline-block', 'padding-left': '11%', 'text-align': 'center', 'fontSize': 36,
                       'color': 'white', 'font-family': 'Raleway'}),
        html.H1("...", style={'fontColor': '#3c3c3c', 'fontSize': 6})
    ], style={'backgroundColor': '#333333'}),
    html.Div([html.A('Refresh', href='/')], style={}),
    html.Div([
        html.H2("Upload Data", style={'fontSize': 24, 'font-family': 'Raleway', 'color': '#333333'}, ),
        html.H3("Upload .txt, .csv or .xls files to starting exploring data...", style={'fontSize': 16,
                                                                                        'font-family': 'Raleway'}),
        dcc.Store(id='csv-data', storage_type='session', data=None),
        html.Div([dcc.Upload(
            id='data-table-upload',
            children=html.Div([html.Button('Upload File')],
                              style={'height': "60px", 'borderWidth': '1px',
                                     'borderRadius': '5px',
                                     'textAlign': 'center',

                                     }),
            multiple=False
        ),
            html.Div(id='output-data-upload'),
        ]), ], style={'display': 'inline-block', 'padding-left': '1%', }),
    html.Div([dcc.Tabs([
        dcc.Tab(label='About', style=tab_style, selected_style=tab_selected_style,
                children=[html.Div([html.H2(" What are AAML's Principal Component Analysis Visualisation Tools?",
                                            style={'fontSize': 18, 'font-family': 'Raleway', 'font-weight': 'bold'
                                                   }),
                                    html.Div([' '.join(MOF_tool_about)]
                                             , style={'font-family': 'Raleway'}),
                                    html.H2(["Scree Plot"],
                                            style={'fontSize': 18,
                                                   'font-family': 'Raleway', 'font-weight': 'bold'}),
                                    html.Div([' '.join(Scree_plot_about)], style={'font-family': 'Raleway'}),
                                    html.H2(["Feature Correlation"], style={'fontSize': 18,
                                                                            'font-weight': 'bold',
                                                                            'font-family': 'Raleway'}),
                                    html.Div([' '.join(Feature_correlation_filter)], style={'font-family': 'Raleway', }),
                                    html.H2(["Plots"],
                                            style={'fontSize': 18, 'font-weight': 'bold',
                                                   'font-family': 'Raleway'}),
                                    html.Div([' '.join(plots_analysis)], style={'font-family': 'Raleway'}),
                                    html.H2(["Data tables"],
                                            style={'fontSize': 18, 'font-weight': 'bold',
                                                   'font-family': 'Raleway'}),
                                    html.Div([' '.join(data_table_download)], style={'font-family': 'Raleway'}),

                                    # ADD LINK
                                    html.Div([html.Plaintext(
                                        [' Click ', html.A('here ',
                                                           href='https://github.com/aaml-analytics/pca-explorer')],
                                        style={'display': 'inline-block',
                                               'fontSize': 14, 'font-family': 'Raleway'}),
                                        html.Div([' '.join(MOF_GH)], style={'display': 'inline-block',
                                                                            'fontSize': 14,
                                                                            'font-family': 'Raleway'}),
                                        html.Img(
                                            src='https://raw.githubusercontent.com/aaml-analytics/mof'
                                                '-explorer/master/github.png',
                                            height='40', width='40',
                                            style={'display': 'inline-block', 'float': "right"
                                                   })
                                    ]
                                        , style={'display': 'inline-block'})
                                    ], style={'backgroundColor': '#ffffff', 'padding-left': '1%'}
                                   )]),
        dcc.Tab(label='Scree Plot', style=tab_style, selected_style=tab_selected_style,
                children=[
                    html.Div([dcc.Graph(id='PC-Eigen-plot')
                              ],
                             style={'display': 'inline-block',
                                    'width': '49%'}),
                    html.Div([dcc.Graph(id='PC-Var-plot')
                              ], style={'display': 'inline-block', 'float': 'right',
                                        'width': '49%'}),
                    html.Div(
                        [html.Label(["Remove outliers (if any) in analysis:", dcc.RadioItems(
                            id='outlier-value',
                            options=[{'label': 'Yes', 'value': 'Yes'},
                                     {'label': 'No', 'value': 'No'}],
                            value='No')
                                     ])
                         ], style={'display': 'inline-block',
                                   'width': '49%', 'padding-left': '1%'}),
                    html.Div([html.Label(["Select the type of matrix used to calculate the principal components:",
                                          dcc.RadioItems(
                                              id='matrix-type-scree',
                                              options=[{'label': 'Correlation', 'value': 'Correlation'},
                                                       {'label': 'Covariance', 'value': 'Covariance'}],
                                              value='Correlation')
                                          ])], style={'display': 'inline-block',
                                                      'width': '49%', }),
                    html.Div([html.P(
                        "Note: Use a correlation matrix when your variables have different scales and you want to weight "
                        "all the variables equally. Use a covariance matrix when your variables have different scales and"
                        " you want to give more emphasis to variables with higher variances. When unsure"
                        " use a correlation matrix.")],
                        style={'padding-left': '1%'}),
                    html.Div([
                        html.Label(["You should attempt to use at least..."

                                       , html.Div(id='var-output-container-filter')])
                    ], style={'padding-left': '1%'}
                    ),
                    html.Div([
                        html.Label(["As a rule of thumb for the Scree Plot"
                                    " Eigenvalues, the point where the slope of the curve "
                                    "is clearly "
                                    "leveling off (the elbow), indicates the number of "
                                    "components that "
                                    "should be retained as significant."])
                    ], style={'padding-left': '1%'}),
                ]),
        dcc.Tab(label='Feature correlation', style=tab_style,
                selected_style=tab_selected_style,
                children=[html.Div([html.Div([dcc.Graph(id='PC-feature-heatmap')
                                              ], style={'width': '47%',
                                                        'display': 'inline-block',
                                                        'float': 'right'}),
                                    html.Div([dcc.Graph(id='feature-heatmap')
                                              ], style={'width': '51%',
                                                        'display': 'inline-block',
                                                        'float': 'left'}),
                                    html.Div([html.Label(["Loading colour bar range:"
                                                             , html.Div(
                                            id='color-range-container')])
                                              ], style={
                                        'fontSize': 12,
                                        'float': 'right',
                                        'width': '100%',
                                        'padding-left': '85%'}
                                             ),
                                    html.Div(
                                        [html.Label(
                                            ["Remove outliers (if any) in analysis:",
                                             dcc.RadioItems(
                                                 id='PC-feature-outlier-value',
                                                 options=[{'label': 'Yes', 'value': 'Yes'},
                                                          {'label': 'No', 'value': 'No'}],
                                                 value='No')
                                             ])
                                        ], style={'display': 'inline-block',
                                                  'width': '29%', 'padding-left': '1%'}),
                                    html.Div([html.Label(
                                        ["Select the type of matrix used to calculate the principal components:",
                                         dcc.RadioItems(
                                             id='matrix-type-heatmap',
                                             options=[{'label': 'Correlation', 'value': 'Correlation'},
                                                      {'label': 'Covariance', 'value': 'Covariance'}],
                                             value='Correlation')
                                         ])], style={'display': 'inline-block',
                                                     'width': '39%', }),
                                    html.Div([html.Label(["Select color scale:",
                                                          dcc.RadioItems(
                                                              id='colorscale',
                                                              options=[{'label': i, 'value': i}
                                                                       for i in
                                                                       ['Viridis', 'Plasma']],
                                                              value='Plasma'
                                                          )]),
                                              ], style={'display': 'inline-block',
                                                        'width': '29%', 'padding-left': '1%'}),
                                    html.Div([html.P(
                                        "Note: Use a correlation matrix when your variables have different scales and you want to weight "
                                        "all the variables equally. Use a covariance matrix when your variables have different scales and"
                                        " you want to give more emphasis to variables with higher variances. When unsure"
                                        " use a correlation matrix.")],
                                        style={'padding-left': '1%'}),
                                    html.Div([
                                        html.P("There are usually two ways multicollinearity, "
                                               "which is when there are a number of variables "
                                               "that are highly correlated, is dealt with:"),
                                        html.P("1) Use PCA to obtain a set of orthogonal ("
                                               "not correlated) variables to analyse."),
                                        html.P("2) Use correlation of determination (R²) to "
                                               "determine which variables are highly "
                                               "correlated and use only 1 in analysis. "
                                               "Cut off for highly correlated variables "
                                               "is ~0.7."),
                                        html.P(
                                            "In any case, it depends on the machine learning algorithm you may apply later. For correlation robust algorithms,"
                                            " such as Random Forest, correlation of features will not be a concern. For non-correlation robust algorithms such as Linear Discriminant Analysis, "
                                            "all high correlation variables should be removed.")

                                    ], style={'padding-left': '1%'}
                                    ),
                                    html.Div([
                                        html.Label(["Note: Data has been standardised (scale)"])
                                    ], style={'padding-left': '1%'})
                                    ])
                          ]),
        dcc.Tab(label='Plots', style=tab_style,
                selected_style=tab_selected_style,
                children=[html.Div([
                    html.Div([html.P("Selecting Features")], style={'padding-left': '1%',
                                                                    'font-weight': 'bold'}),
                    html.Div([
                        html.P("Input here affects all plots, datatables and downloadable data output"),
                        html.Label([
                            "Would you like to analyse all variables or choose custom variables to "
                            "analyse:",
                            dcc.RadioItems(
                                id='all-custom-choice',
                                options=[{'label': 'All',
                                          'value': 'All'},
                                         {'label': 'Custom',
                                          'value': 'Custom'}],
                                value='All'
                            )])
                    ], style={'padding-left': '1%'}),
                    html.Div([
                        html.P("For custom variables input variables you would not like as features in your PCA:"),
                        html.Label(
                            [
                                "Note: Only input numerical variables (non-numerical variables have already "
                                "been removed from your dataframe)",
                                dcc.Dropdown(id='feature-input',
                                             multi=True,
                                             )])
                    ], style={'padding': 10, 'padding-left': '1%'}),
                ]), dcc.Tabs(id='sub-tabs1', style=tabs_styles,
                             children=[
                                 dcc.Tab(label='Biplot (Scores + loadings)', style=tab_style,
                                         selected_style=tab_selected_style,
                                         children=[
                                             html.Div([dcc.Graph(id='biplot', figure=fig)
                                                       ], style={'height': '100%', 'width': '75%',
                                                                 'padding-left': '20%'},
                                                      ),
                                             html.Div(
                                                 [html.Label(
                                                     ["Remove outliers (if any) in analysis:",
                                                      dcc.RadioItems(
                                                          id='outlier-value-biplot',
                                                          options=[
                                                              {'label': 'Yes', 'value': 'Yes'},
                                                              {'label': 'No', 'value': 'No'}],
                                                          value='No')
                                                      ])
                                                 ], style={'display': 'inline-block',
                                                           'width': '29%', 'padding-left': '1%'}),
                                             html.Div([html.Label([
                                                 "Select the type of matrix used to calculate the principal components:",
                                                 dcc.RadioItems(
                                                     id='matrix-type-biplot',
                                                     options=[{'label': 'Correlation',
                                                               'value': 'Correlation'},
                                                              {'label': 'Covariance',
                                                               'value': 'Covariance'}],
                                                     value='Correlation')
                                             ])], style={'display': 'inline-block',
                                                         'width': '39%', }),
                                             html.Div([
                                                 html.Label([
                                                     "Graph Update to show either loadings (Loading Plot) or "
                                                     "scores and loadings (Biplot):",
                                                     dcc.RadioItems(
                                                         id='customvar-graph-update',
                                                         options=[{'label': 'Biplot',
                                                                   'value': 'Biplot'},
                                                                  {'label': 'Loadings',
                                                                   'value': 'Loadings'}],
                                                         value='Biplot')
                                                 ])
                                             ], style={'display': 'inline-block',
                                                       'width': '29%', 'padding-left': '1%'}),
                                             html.Div([html.P(
                                                 "Note: Use a correlation matrix when your variables have different scales and you want to weight "
                                                 "all the variables equally. Use a covariance matrix when your variables have different scales and"
                                                 " you want to give more emphasis to variables with higher variances. When unsure"
                                                 " use a correlation matrix. PCA is an unsupervised machine learning technique - it only "
                                                 "looks at the input features and does not take "
                                                 "into account the output or the target"
                                                 " (response) variable.")],
                                                 style={'padding-left': '1%'}),
                                             html.Div([
                                                 html.P("For variables you have dropped..."),
                                                 html.Label([
                                                     "Would you like to introduce a first target variable"
                                                     " into your data visualisation?"
                                                     " (Graph type must be Biplot): "
                                                     "",
                                                     dcc.RadioItems(
                                                         id='radio-target-item',
                                                         options=[{'label': 'Yes',
                                                                   'value': 'Yes'},
                                                                  {'label': 'No',
                                                                   'value': 'No'}],
                                                         value='No'
                                                     )])
                                             ], style={'width': '49%', 'padding-left': '1%',
                                                       'display': 'inline-block'}),
                                             html.Div([
                                                 html.Label([
                                                     "Select first target variable for color scale of scores: ",
                                                     dcc.Dropdown(
                                                         id='color-scale-scores',
                                                     )])
                                             ], style={'width': '49%', 'padding-left': '1%',
                                                       'display': 'inline-block'}),
                                             html.Div([
                                                 html.Label([
                                                     "Would you like to introduce a second target variable"
                                                     " into your data visualisation??"
                                                     " (Graph type must be Biplot):",
                                                     dcc.RadioItems(
                                                         id='radio-target-item-second',
                                                         options=[{'label': 'Yes',
                                                                   'value': 'Yes'},
                                                                  {'label': 'No',
                                                                   'value': 'No'}],
                                                         value='No'
                                                     )])
                                             ], style={'width': '49%', 'padding-left': '1%',
                                                       'display': 'inline-block'}),
                                             html.Div([
                                                 html.Label([
                                                     "Select second target variable for size scale of scores:",
                                                     dcc.Dropdown(
                                                         id='size-scale-scores',
                                                     )])
                                             ], style={'width': '49%', 'padding-left': '1%',
                                                       'display': 'inline-block'}),
                                             html.Div([html.Label(["Size range:"
                                                                      , html.Div(
                                                     id='size-second-target-container')])
                                                       ], style={'display': 'inline-block',
                                                                 'float': 'right',
                                                                 'padding-right': '5%'}
                                                      ),
                                             html.Div([
                                                 html.Br(),
                                                 html.P(
                                                     "A loading plot shows how "
                                                     "strongly each characteristic (variable)"
                                                     " influences a principal component. The angles between the vectors"
                                                     " tell us how characteristics correlate with one another: "),
                                                 html.P("1) When two vectors are close, forming a small angle, the two "
                                                        "variables they represent are positively correlated. "),
                                                 html.P(
                                                     "2) If they meet each other at 90°, they are not likely to be correlated. "),
                                                 html.P(
                                                     "3) When they diverge and form a large angle (close to 180°), they are negative correlated."),
                                                 html.P(
                                                     "The Score Plot involves the projection of the data onto the PCs in two dimensions."
                                                     "The plot contains the original data but in the rotated (PC) coordinate system"),
                                                 html.P(
                                                     "A biplot merges a score plot and loading plot together.")
                                             ], style={'padding-left': '1%'}
                                             ),

                                         ]),
                                 dcc.Tab(label='Cos2', style=tab_style,
                                         selected_style=tab_selected_style,
                                         children=[
                                             html.Div([dcc.Graph(id='cos2-plot', figure=fig)
                                                       ], style={'width': '65%',
                                                                 'padding-left': '25%'},
                                                      ),
                                             html.Div(
                                                 [html.Label(["Remove outliers (if any) in analysis:",
                                                              dcc.RadioItems(
                                                                  id='outlier-value-cos2',
                                                                  options=[
                                                                      {'label': 'Yes', 'value': 'Yes'},
                                                                      {'label': 'No', 'value': 'No'}],
                                                                  value='No')
                                                              ])
                                                  ], style={'display': 'inline-block',
                                                            'padding-left': '1%',
                                                            'width': '49%'}),
                                             html.Div([html.Label([
                                                 "Select the type of matrix used to calculate the principal components:",
                                                 dcc.RadioItems(
                                                     id='matrix-type-cos2',
                                                     options=[{'label': 'Correlation',
                                                               'value': 'Correlation'},
                                                              {'label': 'Covariance',
                                                               'value': 'Covariance'}],
                                                     value='Correlation')
                                             ])], style={'display': 'inline-block',
                                                         'width': '49%', }),
                                             html.Div([html.P(
                                                 "Note: Use a correlation matrix when your variables have different scales and you want to weight "
                                                 "all the variables equally. Use a covariance matrix when your variables have different scales and"
                                                 " you want to give more emphasis to variables with higher variances. When unsure"
                                                 " use a correlation matrix.")],
                                                 style={'padding-left': '1%'}),
                                             html.Div([
                                                 html.P("The squared cosine shows the importance of a "
                                                        "component for a given observation i.e. "
                                                        "measures "
                                                        " how much a variable is represented in a "
                                                        "component")
                                             ], style={'padding-left': '1%'}),
                                         ]),
                                 dcc.Tab(label='Contribution', style=tab_style,
                                         selected_style=tab_selected_style,
                                         children=[
                                             html.Div([dcc.Graph(id='contrib-plot', figure=fig)
                                                       ], style={'width': '65%',
                                                                 'padding-left': '25%'},
                                                      ),
                                             html.Div(
                                                 [html.Label(["Remove outliers (if any) in analysis:",
                                                              dcc.RadioItems(
                                                                  id='outlier-value-contrib',
                                                                  options=[
                                                                      {'label': 'Yes', 'value': 'Yes'},
                                                                      {'label': 'No', 'value': 'No'}],
                                                                  value='No')
                                                              ], style={'padding-left': '1%'})
                                                  ], style={'display': 'inline-block',
                                                            'width': '49%'}),
                                             html.Div([html.Label([
                                                 "Select the type of matrix used to calculate the principal components:",
                                                 dcc.RadioItems(
                                                     id='matrix-type-contrib',
                                                     options=[{'label': 'Correlation',
                                                               'value': 'Correlation'},
                                                              {'label': 'Covariance',
                                                               'value': 'Covariance'}],
                                                     value='Correlation')
                                             ])], style={'display': 'inline-block',
                                                         'width': '49%', }),
                                             html.Div([html.P(
                                                 "Note: Use a correlation matrix when your variables have different scales and you want to weight "
                                                 "all the variables equally. Use a covariance matrix when your variables have different scales and"
                                                 " you want to give more emphasis to variables with higher variances. When unsure"
                                                 " use a correlation matrix.")],
                                                 style={'padding-left': '1%'}),
                                             html.Div([
                                                 html.P("The contribution plot contains the "
                                                        "contributions (in percentage) of the "
                                                        "variables to the principal components")
                                             ], style={'padding-left': '1%'}),
                                         ])

                             ])
                ]),
        dcc.Tab(label='Data tables', style=tab_style,
                selected_style=tab_selected_style,
                children=[html.Div([
                    html.Div([
                        html.Label(
                            ["Note: Input in 'Plots' tab will provide output of data tables and the"
                             " downloadable PCA data"])
                    ], style={'font-weight': 'bold', 'padding-left': '1%'}),
                    html.Div([html.A(
                        'Download PCA Data (scores for each principal component)',
                        id='download-link',
                        href="",
                        target="_blank"
                    )], style={'padding-left': '1%'}),
                    html.Div([html.Label(["Remove outliers (if any) in analysis:",
                                          dcc.RadioItems(id="eigenA-outlier",
                                                         options=[{'label': 'Yes',
                                                                   'value': 'Yes'},
                                                                  {'label': 'No',
                                                                   'value': 'No'}],
                                                         value='No'
                                                         )])], style={'padding-left': '1%',
                                                                      'display': 'inline-block', 'width': '49%'}),
                    html.Div([html.Label(["Select the type of matrix used to calculate the principal components:",
                                          dcc.RadioItems(
                                              id='matrix-type-data-table',
                                              options=[{'label': 'Correlation', 'value': 'Correlation'},
                                                       {'label': 'Covariance', 'value': 'Covariance'}],
                                              value='Correlation')
                                          ])], style={'display': 'inline-block',
                                                      'width': '49%', }),
                    html.Div([html.P(
                        "Note: Use a correlation matrix when your variables have different scales and you want to weight "
                        "all the variables equally. Use a covariance matrix when your variables have different scales and"
                        " you want to give more emphasis to variables with higher variances. When unsure"
                        " use a correlation matrix.")],
                        style={'padding-left': '1%'}),
                    html.Div([
                        html.Div([
                            html.Label(["Correlation between Features"])
                        ], style={'font-weight': 'bold'}),
                        html.Div([
                            dash_table.DataTable(id='data-table-correlation',
                                                 editable=False,
                                                 filter_action='native',
                                                 sort_action='native',
                                                 sort_mode='multi',
                                                 selected_columns=[],
                                                 selected_rows=[],
                                                 page_action='native',
                                                 column_selectable='single',
                                                 page_current=0,
                                                 page_size=20,
                                                 style_data={'height': 'auto'},
                                                 style_table={'overflowX': 'scroll',
                                                              'maxHeight': '300px',
                                                              'overflowY': 'scroll'},
                                                 style_cell={
                                                     'minWidth': '0px', 'maxWidth': '220px',
                                                     'whiteSpace': 'normal',
                                                 }
                                                 ),
                            html.Div(id='data-table-correlation-container'),
                        ]),
                        html.Div([html.A(
                            'Download Feature Correlation data',
                            id='download-link-correlation',
                            href="",
                            target="_blank"
                        )]),

                    ], style={'padding': 20}),

                    html.Div([
                        html.Div([
                            html.Label(["Eigen Analysis of the correlation matrix"]),
                        ], style={'font-weight': 'bold'}),
                        html.Div([
                            dash_table.DataTable(id='data-table-eigenA',
                                                 editable=False,
                                                 filter_action='native',
                                                 sort_action='native',
                                                 sort_mode='multi',
                                                 selected_columns=[],
                                                 selected_rows=[],
                                                 page_action='native',
                                                 column_selectable='single',
                                                 page_current=0,
                                                 page_size=20,
                                                 style_data={'height': 'auto'},
                                                 style_table={'overflowX': 'scroll',
                                                              'maxHeight': '300px',
                                                              'overflowY': 'scroll'},
                                                 style_cell={
                                                     'minWidth': '0px', 'maxWidth': '220px',
                                                     'whiteSpace': 'normal',
                                                 }
                                                 ),
                            html.Div(id='data-table-eigenA-container'),

                        ]),
                        html.Div([html.A(
                            'Download Eigen Analysis data',
                            id='download-link-eigenA',
                            href="",
                            download='Eigen_Analysis_data.csv',
                            target="_blank"
                        )]),
                    ], style={'padding': 20}),
                    html.Div([
                        html.Div([
                            html.Label(["Loadings (Feature and PC correlation) from PCA"]),
                        ], style={'font-weight': 'bold'}),
                        html.Div([
                            dash_table.DataTable(id='data-table-loadings',
                                                 editable=False,
                                                 filter_action='native',
                                                 sort_action='native',
                                                 sort_mode='multi',
                                                 selected_columns=[],
                                                 selected_rows=[],
                                                 page_action='native',
                                                 column_selectable='single',
                                                 page_current=0,
                                                 page_size=20,
                                                 style_data={'height': 'auto'},
                                                 style_table={'overflowX': 'scroll',
                                                              'maxHeight': '300px',
                                                              'overflowY': 'scroll'},
                                                 style_cell={
                                                     'minWidth': '0px', 'maxWidth': '220px',
                                                     'whiteSpace': 'normal',
                                                 }
                                                 ),
                            html.Div(id='data-table-loadings-container'),
                        ]),
                        html.Div([html.A(
                            'Download Loadings data',
                            id='download-link-loadings',
                            download='Loadings_data.csv',
                            href="",
                            target="_blank"
                        )]),
                    ], style={'padding': 20}),
                    html.Div([
                        html.Div([
                            html.Label(["Cos2 from PCA"])
                        ], style={'font-weight': 'bold'}),
                        html.Div([
                            dash_table.DataTable(id='data-table-cos2',
                                                 editable=False,
                                                 filter_action='native',
                                                 sort_action='native',
                                                 sort_mode='multi',
                                                 selected_columns=[],
                                                 selected_rows=[],
                                                 page_action='native',
                                                 column_selectable='single',
                                                 page_current=0,
                                                 page_size=20,
                                                 style_data={'height': 'auto'},
                                                 style_table={'overflowX': 'scroll',
                                                              'maxHeight': '300px',
                                                              'overflowY': 'scroll'},
                                                 style_cell={
                                                     'minWidth': '0px', 'maxWidth': '220px',
                                                     'whiteSpace': 'normal',
                                                 }
                                                 ),
                            html.Div(id='data-table-cos2-container'),
                        ]),
                        html.Div([html.A(
                            'Download Cos2 data',
                            id='download-link-cos2',
                            download='Cos2_data.csv',
                            href="",
                            target="_blank"
                        )]),
                    ], style={'padding': 20}),
                    html.Div([
                        html.Div([
                            html.Label(["Contributions from PCA"])
                        ], style={'font-weight': 'bold'}),
                        html.Div([
                            dash_table.DataTable(id='data-table-contrib',
                                                 editable=False,
                                                 filter_action='native',
                                                 sort_action='native',
                                                 sort_mode='multi',
                                                 selected_columns=[],
                                                 selected_rows=[],
                                                 page_action='native',
                                                 column_selectable='single',
                                                 page_current=0,
                                                 page_size=20,
                                                 style_data={'height': 'auto'},
                                                 style_table={'overflowX': 'scroll',
                                                              'maxHeight': '300px',
                                                              'overflowY': 'scroll'},
                                                 style_cell={
                                                     'minWidth': '0px', 'maxWidth': '220px',
                                                     'whiteSpace': 'normal',
                                                 }
                                                 ),
                            html.Div(id='data-table-contrib-container'),
                        ]),
                        html.Div([html.A(
                            'Download Contributions data',
                            id='download-link-contrib',
                            download='Contributions_data.csv',
                            href="",
                            target="_blank"
                        )]),
                    ], style={'padding': 20}),
                ])])
    ])
    ], style={'font-family': 'Raleway'})])


# READ FILE
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df.fillna(0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df.fillna(0)
        elif 'txt' or 'tsv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
            df.fillna(0)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df


@app.callback(Output('csv-data', 'data'),
              [Input('data-table-upload', 'contents')],
              [State('data-table-upload', 'filename')])
def parse_uploaded_file(contents, filename):
    if not filename:
        return dash.no_update
    df = parse_contents(contents, filename)
    df.fillna(0)
    return df.to_json(date_format='iso', orient='split')


@app.callback(Output('PC-Var-plot', 'figure'),
              [Input('outlier-value', 'value'),
               Input('matrix-type-scree', 'value'),
               Input('csv-data', 'data')],
              )
def update_graph_stat(outlier, matrix_type, data):
    traces = []
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No' and matrix_type == 'Correlation':
        features1 = dff.columns
        features = list(features1)
        x = dff.loc[:, features].values
        # Separating out the target (if any)
        # Standardizing the features to {mean, variance} = {0, 1}
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=len(features))
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC' + str(i + 1) for i in range(len(features))])
        finalDf = pd.concat([df[[df.columns[0]]], principalDf], axis=1)
        loading = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_df = pd.DataFrame(data=loading[0:, 0:], index=features,
                                  columns=['PC' + str(i + 1) for i in range(loading.shape[1])])
        Var = pca.explained_variance_ratio_
        PC_df = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features))], columns=['Principal Component'])
        Var_df = pd.DataFrame(data=Var, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum = Var_df.cumsum()
        Var_dff = pd.concat([PC_df, (Var_cumsum * 100)], axis=1)
        data = Var_dff
    elif outlier == 'Yes' and matrix_type == 'Correlation':
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        x_outlier = outlier_dff.loc[:, features_outlier].values
        # Standardizing the features
        x_outlier = StandardScaler().fit_transform(x_outlier)

        pca_outlier = PCA(n_components=len(features_outlier))
        principalComponents_outlier = pca_outlier.fit_transform(x_outlier)
        principalDf_outlier = pd.DataFrame(data=principalComponents_outlier
                                           , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier = pd.concat([outlier_names, principalDf_outlier], axis=1)
        # calculating loading
        loading_outlier = pca_outlier.components_.T * np.sqrt(pca_outlier.explained_variance_)
        loading_df_outlier = pd.DataFrame(data=loading_outlier[0:, 0:], index=features_outlier,
                                          columns=['PC' + str(i + 1) for i in range(loading_outlier.shape[1])])
        Var_outlier = pca_outlier.explained_variance_ratio_
        PC_df_outlier = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier))],
                                     columns=['Principal Component'])
        Var_df_outlier = pd.DataFrame(data=Var_outlier, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_outlier = Var_df_outlier.cumsum()
        Var_dff_outlier = pd.concat([PC_df_outlier, (Var_cumsum_outlier * 100)], axis=1)
        data = Var_dff_outlier
    elif outlier == 'No' and matrix_type == 'Covariance':
        features1_covar = dff.columns
        features_covar = list(features1_covar)
        x = dff.loc[:, features_covar].values
        pca_covar = PCA(n_components=len(features_covar))
        principalComponents_covar = pca_covar.fit_transform(x)
        principalDf_covar = pd.DataFrame(data=principalComponents_covar
                                         , columns=['PC' + str(i + 1) for i in range(len(features_covar))])
        finalDf_covar = pd.concat([df[[df.columns[0]]], principalDf_covar], axis=1)
        loading_covar = pca_covar.components_.T * np.sqrt(pca_covar.explained_variance_)
        loading_df_covar = pd.DataFrame(data=loading_covar[0:, 0:], index=features_covar,
                                        columns=['PC' + str(i + 1) for i in range(loading_covar.shape[1])])
        Var_covar = pca_covar.explained_variance_ratio_
        PC_df_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_covar))],
                                   columns=['Principal Component'])
        Var_df_covar = pd.DataFrame(data=Var_covar, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_covar = Var_df_covar.cumsum()
        Var_dff_covar = pd.concat([PC_df_covar, (Var_cumsum_covar * 100)], axis=1)
        data = Var_dff_covar
    elif outlier == 'Yes' and matrix_type == 'Covariance':
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier_covar = outlier_dff.columns
        features_outlier_covar = list(features1_outlier_covar)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]
        x_outlier = outlier_dff.loc[:, features_outlier_covar].values
        pca_outlier_covar = PCA(n_components=len(features_outlier_covar))
        principalComponents_outlier_covar = pca_outlier_covar.fit_transform(x_outlier)
        principalDf_outlier_covar = pd.DataFrame(data=principalComponents_outlier_covar
                                                 , columns=['PC' + str(i + 1) for i in
                                                            range(len(features_outlier_covar))])
        # combining principle components and target
        finalDf_outlier_covar = pd.concat([outlier_names, principalDf_outlier_covar], axis=1)
        # calculating loading
        loading_outlier_covar = pca_outlier_covar.components_.T * np.sqrt(pca_outlier_covar.explained_variance_)
        loading_df_outlier_covar = pd.DataFrame(data=loading_outlier_covar[0:, 0:], index=features_outlier_covar,
                                                columns=['PC' + str(i + 1) for i in
                                                         range(loading_outlier_covar.shape[1])])
        Var_outlier_covar = pca_outlier_covar.explained_variance_ratio_
        PC_df_outlier_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier_covar))],
                                           columns=['Principal Component'])
        Var_df_outlier_covar = pd.DataFrame(data=Var_outlier_covar,
                                            columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_outlier_covar = Var_df_outlier_covar.cumsum()
        Var_dff_outlier_covar = pd.concat([PC_df_outlier_covar, (Var_cumsum_outlier_covar * 100)], axis=1)
        data = Var_dff_outlier_covar
    traces.append(go.Scatter(x=data['Principal Component'], y=data['Cumulative Proportion of Explained Variance'],
                             mode='lines', line=dict(color='Red')))
    return {'data': traces,

            'layout': go.Layout(title='<b>Cumulative Scree Plot Proportion of Explained Variance</b>',
                                titlefont=dict(family='Helvetica', size=16),
                                xaxis={'title': 'Principal Component',
                                       'mirror': True,
                                       'ticks': 'outside',
                                       'showline': True,
                                       'showspikes': True
                                       }, yaxis={'title': 'Cumulative Explained Variance',
                                                 'mirror': True,
                                                 'ticks': 'outside',
                                                 'showline': True,
                                                 'showspikes': True,
                                                 'range': [0, 100]},
                                hovermode='closest', font=dict(family="Helvetica"), template="simple_white")
            }


@app.callback(
    Output('var-output-container-filter', 'children'),
    [Input('outlier-value', 'value'),
     Input('matrix-type-scree', 'value'),
     Input('csv-data', 'data')],
)
def update_output(outlier, matrix_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No' and matrix_type == 'Correlation':
        features1 = dff.columns
        features = list(features1)
        x = dff.loc[:, features].values
        # Separating out the target (if any)
        # Standardizing the features to {mean, variance} = {0, 1}
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=len(features))
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf = pd.concat([df[[df.columns[0]]], principalDf], axis=1)
        dfff = finalDf
        loading = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_df = pd.DataFrame(data=loading[0:, 0:], index=features,
                                  columns=['PC' + str(i + 1) for i in range(loading.shape[1])])
        loading_dff = loading_df.T
        Var = pca.explained_variance_ratio_
        PC_df = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features))], columns=['Principal Component'])
        PC_num = [float(i + 1) for i in range(len(features))]
        Var_df = pd.DataFrame(data=Var, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum = Var_df.cumsum()
        Var_dff = pd.concat([PC_df, (Var_cumsum * 100)], axis=1)
        PC_interp = np.interp(70, Var_dff['Cumulative Proportion of Explained Variance'], PC_num)
        PC_interp_int = math.ceil(PC_interp)
        return "'{}' principal components (≥70% of explained variance) to avoid losing too much of your " \
               "data. Note that there is no required threshold in order for PCA to be valid." \
               " ".format(PC_interp_int)
    elif outlier == 'Yes' and matrix_type == "Correlation":
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        x_outlier = outlier_dff.loc[:, features_outlier].values
        # Separating out the target (if any)
        y_outlier = outlier_dff.loc[:, ].values
        # Standardizing the features
        x_outlier = StandardScaler().fit_transform(x_outlier)

        pca_outlier = PCA(n_components=len(features_outlier))
        principalComponents_outlier = pca_outlier.fit_transform(x_outlier)
        principalDf_outlier = pd.DataFrame(data=principalComponents_outlier
                                           , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier = pd.concat([outlier_names, principalDf_outlier], axis=1)
        dfff_outlier = finalDf_outlier
        # calculating loading
        loading_outlier = pca_outlier.components_.T * np.sqrt(pca_outlier.explained_variance_)
        loading_df_outlier = pd.DataFrame(data=loading_outlier[0:, 0:], index=features_outlier,
                                          columns=['PC' + str(i + 1) for i in range(loading_outlier.shape[1])])
        loading_dff_outlier = loading_df_outlier.T

        Var_outlier = pca_outlier.explained_variance_ratio_
        PC_df_outlier = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier))],
                                     columns=['Principal Component'])
        PC_num_outlier = [float(i + 1) for i in range(len(features_outlier))]
        Var_df_outlier = pd.DataFrame(data=Var_outlier, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_outlier = Var_df_outlier.cumsum()
        Var_dff_outlier = pd.concat([PC_df_outlier, (Var_cumsum_outlier * 100)], axis=1)
        PC_interp_outlier = np.interp(70, Var_dff_outlier['Cumulative Proportion of Explained Variance'],
                                      PC_num_outlier)
        PC_interp_int_outlier = math.ceil(PC_interp_outlier)
        return "'{}' principal components (≥70% of explained variance) to avoid losing too much of your " \
               "data. Note that there is no required threshold in order for PCA to be valid." \
               " ".format(PC_interp_int_outlier)
    elif outlier == 'No' and matrix_type == 'Covariance':
        features1_covar = dff.columns
        features_covar = list(features1_covar)
        x = dff.loc[:, features_covar].values
        pca_covar = PCA(n_components=len(features_covar))
        principalComponents_covar = pca_covar.fit_transform(x)
        principalDf_covar = pd.DataFrame(data=principalComponents_covar
                                         , columns=['PC' + str(i + 1) for i in range(len(features_covar))])
        finalDf_covar = pd.concat([df[[df.columns[0]]], principalDf_covar], axis=1)
        loading_covar = pca_covar.components_.T * np.sqrt(pca_covar.explained_variance_)
        loading_df_covar = pd.DataFrame(data=loading_covar[0:, 0:], index=features_covar,
                                        columns=['PC' + str(i + 1) for i in range(loading_covar.shape[1])])
        Var_covar = pca_covar.explained_variance_ratio_
        PC_df_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_covar))],
                                   columns=['Principal Component'])
        PC_num_covar = [float(i + 1) for i in range(len(features_covar))]
        Var_df_covar = pd.DataFrame(data=Var_covar, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_covar = Var_df_covar.cumsum()
        Var_dff_covar = pd.concat([PC_df_covar, (Var_cumsum_covar * 100)], axis=1)
        PC_interp_covar = np.interp(70, Var_dff_covar['Cumulative Proportion of Explained Variance'], PC_num_covar)
        PC_interp_int_covar = math.ceil(PC_interp_covar)
        return "'{}' principal components (≥70% of explained variance) to avoid losing too much of your " \
               "data. Note that there is no required threshold in order for PCA to be valid." \
               " ".format(PC_interp_int_covar)
    elif outlier == 'Yes' and matrix_type == 'Covariance':
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier_covar = outlier_dff.columns
        features_outlier_covar = list(features1_outlier_covar)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]
        x_outlier = outlier_dff.loc[:, features_outlier_covar].values
        pca_outlier_covar = PCA(n_components=len(features_outlier_covar))
        principalComponents_outlier_covar = pca_outlier_covar.fit_transform(x_outlier)
        principalDf_outlier_covar = pd.DataFrame(data=principalComponents_outlier_covar
                                                 , columns=['PC' + str(i + 1) for i in
                                                            range(len(features_outlier_covar))])
        # combining principle components and target
        finalDf_outlier_covar = pd.concat([outlier_names, principalDf_outlier_covar], axis=1)
        # calculating loading
        loading_outlier_covar = pca_outlier_covar.components_.T * np.sqrt(pca_outlier_covar.explained_variance_)
        loading_df_outlier_covar = pd.DataFrame(data=loading_outlier_covar[0:, 0:], index=features_outlier_covar,
                                                columns=['PC' + str(i + 1) for i in
                                                         range(loading_outlier_covar.shape[1])])
        Var_outlier_covar = pca_outlier_covar.explained_variance_ratio_
        PC_df_outlier_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier_covar))],
                                           columns=['Principal Component'])
        PC_num_outlier_covar = [float(i + 1) for i in range(len(features_outlier_covar))]
        Var_df_outlier_covar = pd.DataFrame(data=Var_outlier_covar,
                                            columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_outlier_covar = Var_df_outlier_covar.cumsum()
        Var_dff_outlier_covar = pd.concat([PC_df_outlier_covar, (Var_cumsum_outlier_covar * 100)], axis=1)
        PC_interp_outlier = np.interp(70, Var_dff_outlier_covar['Cumulative Proportion of Explained Variance'],
                                      PC_num_outlier_covar)
        PC_interp_int_outlier = math.ceil(PC_interp_outlier)
        return "'{}' principal components (≥70% of explained variance) to avoid losing too much of your " \
               "data. Note that there is no required threshold in order for PCA to be valid." \
               " ".format(PC_interp_int_outlier)


@app.callback(Output('PC-Eigen-plot', 'figure'),
              [Input('outlier-value', 'value'),
               Input('matrix-type-scree', 'value'),
               Input('csv-data', 'data')]
              )
def update_graph_stat(outlier, matrix_type, data):
    traces = []
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No' and matrix_type == "Correlation":
        features1 = dff.columns
        features = list(features1)
        x = dff.loc[:, features].values
        # Separating out the target (if any)
        y = dff.loc[:, ].values
        # Standardizing the features to {mean, variance} = {0, 1}
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=len(features))
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf = pd.concat([df[[df.columns[0]]], principalDf], axis=1)
        dfff = finalDf
        loading = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_df = pd.DataFrame(data=loading[0:, 0:], index=features,
                                  columns=['PC' + str(i + 1) for i in range(loading.shape[1])])
        loading_dff = loading_df.T
        Var = pca.explained_variance_ratio_
        PC_df = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features))], columns=['Principal Component'])
        PC_num = [float(i + 1) for i in range(len(features))]
        Var_df = pd.DataFrame(data=Var, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum = Var_df.cumsum()
        Var_dff = pd.concat([PC_df, (Var_cumsum * 100)], axis=1)
        PC_interp = np.interp(70, Var_dff['Cumulative Proportion of Explained Variance'], PC_num)
        PC_interp_int = math.ceil(PC_interp)
        eigenvalues = pca.explained_variance_
        Eigen_df = pd.DataFrame(data=eigenvalues, columns=['Eigenvalues'])
        Eigen_dff = pd.concat([PC_df, Eigen_df], axis=1)
        data = Eigen_dff
    elif outlier == 'Yes' and matrix_type == "Correlation":
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        x_outlier = outlier_dff.loc[:, features_outlier].values
        # Separating out the target (if any)
        y_outlier = outlier_dff.loc[:, ].values
        # Standardizing the features
        x_outlier = StandardScaler().fit_transform(x_outlier)

        pca_outlier = PCA(n_components=len(features_outlier))
        principalComponents_outlier = pca_outlier.fit_transform(x_outlier)
        principalDf_outlier = pd.DataFrame(data=principalComponents_outlier
                                           , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier = pd.concat([outlier_names, principalDf_outlier], axis=1)
        dfff_outlier = finalDf_outlier
        # calculating loading
        loading_outlier = pca_outlier.components_.T * np.sqrt(pca_outlier.explained_variance_)
        loading_df_outlier = pd.DataFrame(data=loading_outlier[0:, 0:], index=features_outlier,
                                          columns=['PC' + str(i + 1) for i in range(loading_outlier.shape[1])])
        loading_dff_outlier = loading_df_outlier.T

        Var_outlier = pca_outlier.explained_variance_ratio_
        PC_df_outlier = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier))],
                                     columns=['Principal Component'])
        PC_num_outlier = [float(i + 1) for i in range(len(features_outlier))]
        Var_df_outlier = pd.DataFrame(data=Var_outlier, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_outlier = Var_df_outlier.cumsum()
        Var_dff_outlier = pd.concat([PC_df_outlier, (Var_cumsum_outlier * 100)], axis=1)
        PC_interp_outlier = np.interp(70, Var_dff_outlier['Cumulative Proportion of Explained Variance'],
                                      PC_num_outlier)
        PC_interp_int_outlier = math.ceil(PC_interp_outlier)
        eigenvalues_outlier = pca_outlier.explained_variance_
        Eigen_df_outlier = pd.DataFrame(data=eigenvalues_outlier, columns=['Eigenvalues'])
        Eigen_dff_outlier = pd.concat([PC_df_outlier, Eigen_df_outlier], axis=1)
        data = Eigen_dff_outlier
    elif outlier == 'No' and matrix_type == "Covariance":
        features1_covar = dff.columns
        features_covar = list(features1_covar)
        x = dff.loc[:, features_covar].values
        pca_covar = PCA(n_components=len(features_covar))
        principalComponents_covar = pca_covar.fit_transform(x)
        principalDf_covar = pd.DataFrame(data=principalComponents_covar
                                         , columns=['PC' + str(i + 1) for i in range(len(features_covar))])
        finalDf_covar = pd.concat([df[[df.columns[0]]], principalDf_covar], axis=1)
        loading_covar = pca_covar.components_.T * np.sqrt(pca_covar.explained_variance_)
        loading_df_covar = pd.DataFrame(data=loading_covar[0:, 0:], index=features_covar,
                                        columns=['PC' + str(i + 1) for i in range(loading_covar.shape[1])])
        Var_covar = pca_covar.explained_variance_ratio_
        PC_df_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_covar))],
                                   columns=['Principal Component'])
        PC_num_covar = [float(i + 1) for i in range(len(features_covar))]
        Var_df_covar = pd.DataFrame(data=Var_covar, columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_covar = Var_df_covar.cumsum()
        Var_dff_covar = pd.concat([PC_df_covar, (Var_cumsum_covar * 100)], axis=1)
        PC_interp_covar = np.interp(70, Var_dff_covar['Cumulative Proportion of Explained Variance'], PC_num_covar)
        PC_interp_int_covar = math.ceil(PC_interp_covar)
        eigenvalues_covar = pca_covar.explained_variance_
        Eigen_df_covar = pd.DataFrame(data=eigenvalues_covar, columns=['Eigenvalues'])
        Eigen_dff_covar = pd.concat([PC_df_covar, Eigen_df_covar], axis=1)
        data = Eigen_dff_covar
    elif outlier == 'Yes' and matrix_type == 'Covariance':
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier_covar = outlier_dff.columns
        features_outlier_covar = list(features1_outlier_covar)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]
        x_outlier = outlier_dff.loc[:, features_outlier_covar].values
        pca_outlier_covar = PCA(n_components=len(features_outlier_covar))
        principalComponents_outlier_covar = pca_outlier_covar.fit_transform(x_outlier)
        principalDf_outlier_covar = pd.DataFrame(data=principalComponents_outlier_covar
                                                 , columns=['PC' + str(i + 1) for i in
                                                            range(len(features_outlier_covar))])
        # combining principle components and target
        finalDf_outlier_covar = pd.concat([outlier_names, principalDf_outlier_covar], axis=1)
        # calculating loading
        loading_outlier_covar = pca_outlier_covar.components_.T * np.sqrt(pca_outlier_covar.explained_variance_)
        loading_df_outlier_covar = pd.DataFrame(data=loading_outlier_covar[0:, 0:], index=features_outlier_covar,
                                                columns=['PC' + str(i + 1) for i in
                                                         range(loading_outlier_covar.shape[1])])
        Var_outlier_covar = pca_outlier_covar.explained_variance_ratio_
        PC_df_outlier_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier_covar))],
                                           columns=['Principal Component'])
        PC_num_outlier_covar = [float(i + 1) for i in range(len(features_outlier_covar))]
        Var_df_outlier_covar = pd.DataFrame(data=Var_outlier_covar,
                                            columns=['Cumulative Proportion of Explained Variance'])
        Var_cumsum_outlier_covar = Var_df_outlier_covar.cumsum()
        Var_dff_outlier_covar = pd.concat([PC_df_outlier_covar, (Var_cumsum_outlier_covar * 100)], axis=1)
        PC_interp_outlier = np.interp(70, Var_dff_outlier_covar['Cumulative Proportion of Explained Variance'],
                                      PC_num_outlier_covar)
        PC_interp_int_outlier = math.ceil(PC_interp_outlier)
        eigenvalues_outlier_covar = pca_outlier_covar.explained_variance_
        Eigen_df_outlier_covar = pd.DataFrame(data=eigenvalues_outlier_covar, columns=['Eigenvalues'])
        Eigen_dff_outlier_covar = pd.concat([PC_df_outlier_covar, Eigen_df_outlier_covar], axis=1)
        data = Eigen_dff_outlier_covar

    traces.append(go.Scatter(x=data['Principal Component'], y=data['Eigenvalues'], mode='lines'))
    return {'data': traces,

            'layout': go.Layout(title='<b>Scree Plot Eigenvalues</b>', xaxis={'title': 'Principal Component',
                                                                              'mirror': True,
                                                                              'ticks': 'outside',
                                                                              'showline': True,
                                                                              'showspikes': True},
                                titlefont=dict(family='Helvetica', size=16),
                                yaxis={'title': 'Eigenvalues', 'mirror': True,
                                           'ticks': 'outside',
                                           'showline': True,
                                           'showspikes': True}, hovermode='closest',
                                font=dict(family="Helvetica"), template="simple_white", )
            }


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


@app.callback([Output('PC-feature-heatmap', 'figure'),
Output('color-range-container', 'children')],
              [
                  Input('PC-feature-outlier-value', 'value'),
                  Input('colorscale', 'value'),
                  Input("matrix-type-heatmap", "value"),
                  Input('csv-data', 'data')]
              )
def update_graph_stat(outlier, colorscale, matrix_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    traces = []
    # INCLUDING OUTLIERS
    features1 = dff.columns
    features = list(features1)
    x = dff.loc[:, features].values
    # Separating out the target (if any)
    y = dff.loc[:, ].values
    # Standardizing the features to {mean, variance} = {0, 1}
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=len(features))
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['PC' + str(i + 1) for i in range(len(features))])
    # combining principle components and target
    finalDf = pd.concat([df[[df.columns[0]]], principalDf], axis=1)
    dfff = finalDf
    # explained variance of the the two principal components
    # print(pca.explained_variance_ratio_)
    # Explained variance tells us how much information (variance) can be attributed to each of the principal components
    # loading of each feature in principle components
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(data=loading[0:, 0:], index=features,
                              columns=['PC' + str(i + 1) for i in range(loading.shape[1])])
    loading_dff = loading_df.T
    # OUTLIERS REMOVED
    z_scores_hm = scipy.stats.zscore(dff)
    abs_z_scores_hm = np.abs(z_scores_hm)
    filtered_entries_hm = (abs_z_scores_hm < 3).all(axis=1)
    outlier_dff_hm = dff[filtered_entries_hm]
    features1_outlier_hm = outlier_dff_hm.columns
    features_outlier2 = list(features1_outlier_hm)
    outlier_names1_hm = df[filtered_entries_hm]
    outlier_names_hm = outlier_names1_hm.iloc[:, 0]
    x_outlier_hm = outlier_dff_hm.loc[:, features_outlier2].values
    # Separating out the target (if any)
    # Standardizing the features
    x_outlier_hm = StandardScaler().fit_transform(x_outlier_hm)

    pca_outlier_hm = PCA(n_components=len(features_outlier2))
    principalComponents_outlier_hm = pca_outlier_hm.fit_transform(x_outlier_hm)
    principalDf_outlier_hm = pd.DataFrame(data=principalComponents_outlier_hm
                                          , columns=['PC' + str(i + 1) for i in range(len(features_outlier2))])
    # combining principle components and target
    finalDf_outlier_hm = pd.concat([outlier_names_hm, principalDf_outlier_hm], axis=1)
    dfff_outlier_hm = finalDf_outlier_hm
    # calculating loading
    loading_outlier_hm = pca_outlier_hm.components_.T * np.sqrt(pca_outlier_hm.explained_variance_)
    loading_df_outlier_hm = pd.DataFrame(data=loading_outlier_hm[0:, 0:], index=features_outlier2,
                                         columns=['PC' + str(i + 1) for i in range(loading_outlier_hm.shape[1])])
    loading_dff_outlier_hm = loading_df_outlier_hm.T
    # COVAR MATRIX
    features1_covar = dff.columns
    features_covar = list(features1_covar)
    x = dff.loc[:, features_covar].values
    pca_covar = PCA(n_components=len(features_covar))
    principalComponents_covar = pca_covar.fit_transform(x)
    principalDf_covar = pd.DataFrame(data=principalComponents_covar
                                     , columns=['PC' + str(i + 1) for i in range(len(features_covar))])
    finalDf_covar = pd.concat([df[[df.columns[0]]], principalDf_covar], axis=1)
    loading_covar = pca_covar.components_.T * np.sqrt(pca_covar.explained_variance_)
    loading_df_covar = pd.DataFrame(data=loading_covar[0:, 0:], index=features_covar,
                                    columns=['PC' + str(i + 1) for i in range(loading_covar.shape[1])])
    loading_dff_covar = loading_df_covar.T
    # COVAR MATRIX OUTLIERS REMOVED

    if outlier == 'No' and matrix_type == "Correlation":
        data = loading_dff
    elif outlier == 'Yes' and matrix_type == "Correlation":
        data = loading_dff_outlier_hm
    elif outlier == 'No' and matrix_type == "Covariance":
        data = loading_dff_covar
    elif outlier == "Yes" and matrix_type == "Covariance":
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier_covar = outlier_dff.columns
        features_outlier_covar = list(features1_outlier_covar)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]
        x_outlier = outlier_dff.loc[:, features_outlier_covar].values
        pca_outlier_covar = PCA(n_components=len(features_outlier_covar))
        principalComponents_outlier_covar = pca_outlier_covar.fit_transform(x_outlier)
        principalDf_outlier_covar = pd.DataFrame(data=principalComponents_outlier_covar
                                                 , columns=['PC' + str(i + 1) for i in
                                                            range(len(features_outlier_covar))])
        # combining principle components and target
        finalDf_outlier_covar = pd.concat([outlier_names, principalDf_outlier_covar], axis=1)
        # calculating loading
        loading_outlier_covar = pca_outlier_covar.components_.T * np.sqrt(pca_outlier_covar.explained_variance_)
        loading_df_outlier_covar = pd.DataFrame(data=loading_outlier_covar[0:, 0:], index=features_outlier_covar,
                                                columns=['PC' + str(i + 1) for i in
                                                         range(loading_outlier_covar.shape[1])])
        loading_dff_outlier_covar = loading_df_outlier_covar.T
        data = loading_dff_outlier_covar
    size_range = [round_up(data.values.min(), 2),round_down(data.values.max(),2) ]
    traces.append(go.Heatmap(
        z=data, x=features_outlier2, y=['PC' + str(i + 1) for i in range(loading_outlier_hm.shape[1])],
        colorscale="Viridis" if colorscale == 'Viridis' else "Plasma",
        # coord: represent the correlation between the various feature and the principal component itself
        colorbar={"title": "Loading",
                  # 'tickvals': [round_up(data.values.min(), 2),
                  #  round_up((data.values.min() + (data.values.max() + data.values.min())/2)/2, 2),
                  #  round_down((data.values.max() + data.values.min())/2,2),
                  #  round_down((data.values.max() + (data.values.max() + data.values.min())/2)/2, 2),
                  #  round_down(data.values.max(),2), ]
                  }
    ))
    return {'data': traces,
            'layout': go.Layout(title=dict(text='<b>PC and Feature Correlation Analysis</b>'),
                                xaxis=dict(title_text='Features', title_standoff=50),
                                titlefont=dict(family='Helvetica', size=16),
                                hovermode='closest', margin={'b': 110, 't': 50, 'l': 75},
                                font=dict(family="Helvetica", size=11),
                                annotations=[
                                    dict(x=-0.16, y=0.5, showarrow=False, text="Principal Components",
                                         xref='paper', yref='paper', textangle=-90,
                                         font=dict(size=12))]
                                ),

            }, '{}'.format(size_range)


@app.callback(Output('feature-heatmap', 'figure'),
              [
                  Input('PC-feature-outlier-value', 'value'),
                  Input('colorscale', 'value'),
                  Input('csv-data', 'data')])
def update_graph_stat(outlier, colorscale, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    traces = []
    if outlier == 'No':
        features1 = dff.columns
        features = list(features1)
        # correlation coefficient and coefficient of determination
        correlation_dff = dff.corr(method='pearson', )
        r2_dff = correlation_dff * correlation_dff
        data = r2_dff
        feat = features
    elif outlier == 'Yes':
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        # correlation coefficient and coefficient of determination
        correlation_dff_outlier = outlier_dff.corr(method='pearson', )
        r2_dff_outlier = correlation_dff_outlier * correlation_dff_outlier
        data = r2_dff_outlier
        feat = features_outlier
    traces.append(go.Heatmap(
        z=data, x=feat, y=feat, colorscale="Viridis" if colorscale == 'Viridis' else "Plasma",
        # coord: represent the correlation between the various feature and the principal component itself
        colorbar={"title": "R²", 'tickvals': [0, 0.2, 0.4, 0.6, 0.8, 1]}))
    return {'data': traces,
            'layout': go.Layout(title=dict(text='<b>Feature Correlation Analysis</b>', y=0.97, x=0.6),
                                xaxis={},
                                titlefont=dict(family='Helvetica', size=16),
                                yaxis={},
                                hovermode='closest', margin={'b': 110, 't': 50, 'l': 180, 'r': 50},
                                font=dict(family="Helvetica", size=11)),
            }


@app.callback(Output('feature-input', 'options'),
              [Input('all-custom-choice', 'value'),
               Input('csv-data', 'data')])
def activate_input(all_custom, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        options = []
    elif all_custom == 'Custom':
        options = [{'label': i, 'value': i} for i in dff.columns]
    return options


@app.callback(Output('color-scale-scores', 'options'),
              [Input('feature-input', 'value'),
               Input('radio-target-item', 'value'),
               Input('outlier-value-biplot', 'value'),
               Input('customvar-graph-update', 'value'),
               Input('matrix-type-biplot', 'value'),
               Input('csv-data', 'data')], )
def populate_color_dropdown(input, target, outlier, graph_type, matrix_type, data):
    if not data:
        return dash.no_update
    if input is None:
        raise dash.exceptions.PreventUpdate
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    dff_target = dff[input]
    z_scores_target = scipy.stats.zscore(dff_target)
    abs_z_scores_target = np.abs(z_scores_target)
    filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
    dff_target_outlier = dff_target[filtered_entries_target]
    if target == 'Yes' and outlier == 'Yes' and matrix_type == "Correlation" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target_outlier.columns]
    elif target == 'Yes' and outlier == 'Yes' and matrix_type == "Covariance" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target_outlier.columns]
    elif target == 'Yes' and outlier == 'No' and matrix_type == "Correlation" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target.columns]
    elif target == 'Yes' and outlier == 'No' and matrix_type == "Covariance" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target.columns]
    elif target == 'No' or graph_type == 'Loadings':
        options = []
    return options


@app.callback(Output('size-scale-scores', 'options'),
              [Input('feature-input', 'value'),
               Input('radio-target-item-second', 'value'),
               Input('outlier-value-biplot', 'value'),
               Input('customvar-graph-update', 'value'),
               Input('matrix-type-biplot', 'value'),
               Input('csv-data', 'data')])
def populate_color_dropdown(input, target, outlier, graph_type, matrix_type, data):
    if not data:
        return dash.no_update
    if input is None:
        raise dash.exceptions.PreventUpdate
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    dff_target = dff[input]
    z_scores_target = scipy.stats.zscore(dff_target)
    abs_z_scores_target = np.abs(z_scores_target)
    filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
    dff_target_outlier = dff_target[filtered_entries_target]
    if target == 'Yes' and outlier == 'Yes' and matrix_type == "Correlation" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target_outlier.columns]
    elif target == 'Yes' and outlier == 'Yes' and matrix_type == "Covariance" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target_outlier.columns]
    elif target == 'Yes' and outlier == 'No' and matrix_type == "Correlation" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target.columns]
    elif target == 'Yes' and outlier == 'No' and matrix_type == "Covariance" and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target.columns]
    elif target == 'No' or graph_type == 'Loadings':
        options = []
    return options


# resume covar matrix...
@app.callback(Output('biplot', 'figure'),
              [
                  Input('outlier-value-biplot', 'value'),
                  Input('feature-input', 'value'),
                  Input('customvar-graph-update', 'value'),
                  Input('color-scale-scores', 'value'),
                  Input('radio-target-item', 'value'),
                  Input('size-scale-scores', 'value'),
                  Input('radio-target-item-second', 'value'),
                  Input('all-custom-choice', 'value'),
                  Input('matrix-type-biplot', 'value'),
                  Input('csv-data', 'data')
              ]
              )
def update_graph_custom(outlier, input, graph_update, color, target, size, target2, all_custom, matrix_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    features1 = dff.columns
    features = list(features1)
    if all_custom == 'All':
        # x_scale = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(x_scale)
        # OUTLIER DATA
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min

        # ORIGINAL DATA WITH OUTLIERS
        x_scale = dff.loc[:, features].values
        y_scale = dff.loc[:, ].values
        x_scale = StandardScaler().fit_transform(x_scale)
        # x_scale = rescale(x_scale, new_min=0, new_max=1)
        pca_scale = PCA(n_components=len(features))
        principalComponents_scale = pca_scale.fit_transform(x_scale)
        principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                         , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale = pd.concat([df[[df.columns[0]]], principalDf_scale], axis=1)
        dfff_scale = finalDf_scale.fillna(0)
        Var_scale = pca_scale.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale = pca_scale.components_.T * np.sqrt(pca_scale.explained_variance_)
        loading_scale_df = pd.DataFrame(data=loading_scale[:, 0:2],
                                        columns=["PC1", "PC2"])
        line_group_scale_df = pd.DataFrame(data=features, columns=['line_group'])
        loading_scale_dff = pd.concat([loading_scale_df, line_group_scale_df], axis=1)
        a = (len(features), 2)
        zero_scale = np.zeros(a)
        zero_scale_df = pd.DataFrame(data=zero_scale, columns=["PC1", "PC2"])
        zero_scale_dff = pd.concat([zero_scale_df, line_group_scale_df], axis=1)
        loading_scale_line_graph = pd.concat([loading_scale_dff, zero_scale_dff], axis=0)

        # ORIGINAL DATA WITH REMOVING OUTLIERS
        x_outlier_scale = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale = outlier_dff.loc[:, ].values
        x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)

        # x_outlier_scale = MinMaxScaler().fit_transform(x_outlier_scale)

        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min

        # x_outlier_scale = rescale(x_outlier_scale, new_min=0, new_max=1)
        # uses covariance matrix
        pca_outlier_scale = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
        principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                 , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier_scale = pd.concat([outlier_names, principalDf_outlier_scale], axis=1)
        dfff_outlier_scale = finalDf_outlier_scale.fillna(0)
        # calculating loading
        Var_outlier_scale = pca_outlier_scale.explained_variance_ratio_
        # calculating loading vector plot
        loading_outlier_scale = pca_outlier_scale.components_.T * np.sqrt(pca_outlier_scale.explained_variance_)
        loading_outlier_scale_df = pd.DataFrame(data=loading_outlier_scale[:, 0:2],
                                                columns=["PC1", "PC2"])
        line_group_df = pd.DataFrame(data=features_outlier, columns=['line_group'])
        loading_outlier_scale_dff = pd.concat([loading_outlier_scale_df, line_group_df], axis=1)
        a = (len(features_outlier), 2)
        zero_outlier_scale = np.zeros(a)
        zero_outlier_scale_df = pd.DataFrame(data=zero_outlier_scale, columns=["PC1", "PC2"])
        zero_outlier_scale_dff = pd.concat([zero_outlier_scale_df, line_group_df], axis=1)
        loading_outlier_scale_line_graph = pd.concat([loading_outlier_scale_dff, zero_outlier_scale_dff], axis=0)
        # COVARIANCE MATRIX
        x_scale_covar = dff.loc[:, features].values
        y_scale_covar = dff.loc[:, ].values
        pca_scale_covar = PCA(n_components=len(features))
        principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
        principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                               , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_covar], axis=1)
        dfff_scale_covar = finalDf_scale_covar.fillna(0)
        Var_scale_covar = pca_scale_covar.explained_variance_ratio_
        loading_scale_covar = pca_scale_covar.components_.T * np.sqrt(pca_scale_covar.explained_variance_)
        loading_scale_df_covar = pd.DataFrame(data=loading_scale_covar[:, 0:2],
                                              columns=["PC1", "PC2"])
        line_group_scale_df_covar = pd.DataFrame(data=features, columns=['line_group'])
        loading_scale_dff_covar = pd.concat([loading_scale_df_covar, line_group_scale_df_covar], axis=1)
        a = (len(features), 2)
        zero_scale_covar = np.zeros(a)
        zero_scale_df_covar = pd.DataFrame(data=zero_scale_covar, columns=["PC1", "PC2"])
        zero_scale_dff_covar = pd.concat([zero_scale_df_covar, line_group_scale_df_covar], axis=1)
        loading_scale_line_graph_covar = pd.concat([loading_scale_dff_covar, zero_scale_dff_covar], axis=0)
        # COVARIANCE MATRIX OUTLIERS
        x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale_covar = outlier_dff.loc[:, ].values
        pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
        principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_outlier))])
        finalDf_outlier_scale_covar = pd.concat([outlier_names, principalDf_outlier_scale_covar], axis=1)
        dfff_outlier_scale_covar = finalDf_outlier_scale_covar.fillna(0)
        Var_outlier_scale_covar = pca_outlier_scale_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_outlier_scale_covar = pca_outlier_scale_covar.components_.T * np.sqrt(
            pca_outlier_scale_covar.explained_variance_)
        loading_outlier_scale_df_covar = pd.DataFrame(data=loading_outlier_scale_covar[:, 0:2],
                                                      columns=["PC1", "PC2"])
        line_group_df_covar = pd.DataFrame(data=features_outlier, columns=['line_group'])
        loading_outlier_scale_dff_covar = pd.concat([loading_outlier_scale_df_covar, line_group_df_covar], axis=1)
        a = (len(features_outlier), 2)
        zero_outlier_scale_covar = np.zeros(a)
        zero_outlier_scale_df_covar = pd.DataFrame(data=zero_outlier_scale_covar, columns=["PC1", "PC2"])
        zero_outlier_scale_dff_covar = pd.concat([zero_outlier_scale_df_covar, line_group_df_covar], axis=1)
        loading_outlier_scale_line_graph_covar = pd.concat(
            [loading_outlier_scale_dff_covar, zero_outlier_scale_dff_covar], axis=0)

        if outlier == 'No' and matrix_type == "Correlation":
            dat = dfff_scale
        elif outlier == 'Yes' and matrix_type == "Correlation":
            dat = dfff_outlier_scale
        elif outlier == "No" and matrix_type == "Covariance":
            dat = dfff_scale_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dat = dfff_outlier_scale_covar
        trace2_all = go.Scatter(x=dat['PC1'], y=dat['PC2'], mode='markers',
                                text=dat[dat.columns[0]],
                                hovertemplate=
                                '<b>%{text}</b>' +
                                '<br>PC1: %{x}<br>' +
                                'PC2: %{y}'
                                "<extra></extra>",

                                marker=dict(opacity=0.7, showscale=False, size=12,
                                            line=dict(width=0.5, color='DarkSlateGrey'),
                                            ),
                                )
        ####################################################################################################
        # INCLUDE THIS
        if outlier == 'No' and matrix_type == "Correlation":
            data = loading_scale_line_graph
            variance = Var_scale
        elif outlier == 'Yes' and matrix_type == "Correlation":
            data = loading_outlier_scale_line_graph
            variance = Var_outlier_scale
        elif outlier == "No" and matrix_type == "Covariance":
            data = loading_scale_line_graph_covar
            variance = Var_scale_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            data = loading_outlier_scale_line_graph_covar
            variance = Var_outlier_scale_covar
        counter = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf_all = data[data['line_group'] == i]
            trace1_all = go.Scatter(x=dataf_all['PC1'], y=dataf_all['PC2'], line=dict(color="#4f4f4f"),
                                    name=i,
                                    # text=i,
                                    meta=i,
                                    hovertemplate=
                                    '<b>%{meta}</b>' +
                                    '<br>PC1: %{x}<br>' +
                                    'PC2: %{y}'
                                    "<extra></extra>",
                                    mode='lines+text',
                                    textposition='bottom right', textfont=dict(size=12)
                                    )
            lists[counter] = trace1_all
            counter = counter + 1
        ####################################################################################################
        if graph_update == 'Biplot':
            lists.insert(0, trace2_all)
            return {'data': lists,
                    'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2))),
                                        yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2))),
                                        showlegend=False, margin={'r': 0},
                                        # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                        #              y0=-1, x1=1, y1=1,
                                        #              line_color="DarkSlateGrey")]
                                        ),

                    }
        elif graph_update == 'Loadings':
            return {'data': lists,
                    'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2))),
                                        yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2))),
                                        showlegend=False, margin={'r': 0},
                                        # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                        #              y0=-1, x1=1, y1=1,
                                        #              line_color="DarkSlateGrey")]
                                        ),

                    }

    elif all_custom == 'Custom':
        # Dropping Data variables
        dff_input = dff.drop(columns=dff[input])
        features1_input = dff_input.columns
        features_input = list(features1_input)
        dff_target = dff[input]
        # OUTLIER DATA INPUT
        z_scores_input = scipy.stats.zscore(dff_input)
        abs_z_scores_input = np.abs(z_scores_input)
        filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
        dff_input_outlier = dff_input[filtered_entries_input]
        features1_input_outlier = dff_input_outlier.columns
        features_input_outlier = list(features1_input_outlier)
        outlier_names_input1 = df[filtered_entries_input]
        outlier_names_input = outlier_names_input1.iloc[:, 0]
        # OUTLIER DATA TARGET
        z_scores_target = scipy.stats.zscore(dff_target)
        abs_z_scores_target = np.abs(z_scores_target)
        filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
        dff_target_outlier = dff_target[filtered_entries_target]
        # INPUT DATA WITH OUTLIERS
        x_scale_input = dff_input.loc[:, features_input].values
        y_scale_input = dff_input.loc[:, ].values
        x_scale_input = StandardScaler().fit_transform(x_scale_input)

        # x_scale_input = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(x_scale_input)
        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min

        # x_scale_input = rescale(x_scale_input, new_min=0, new_max=1)

        pca_scale_input = PCA(n_components=len(features_input))
        principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
        principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                               , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
        dfff_scale_input = finalDf_scale_input.fillna(0)
        Var_scale_input = pca_scale_input.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input = pca_scale_input.components_.T * np.sqrt(pca_scale_input.explained_variance_)
        loading_scale_input_df = pd.DataFrame(data=loading_scale_input[:, 0:2],
                                              columns=["PC1", "PC2"])
        line_group_scale_input_df = pd.DataFrame(data=features_input, columns=['line_group'])
        loading_scale_input_dff = pd.concat([loading_scale_input_df, line_group_scale_input_df],
                                            axis=1)
        a = (len(features_input), 2)
        zero_scale_input = np.zeros(a)
        zero_scale_input_df = pd.DataFrame(data=zero_scale_input, columns=["PC1", "PC2"])
        zero_scale_input_dff = pd.concat([zero_scale_input_df, line_group_scale_input_df], axis=1)
        loading_scale_input_line_graph = pd.concat([loading_scale_input_dff, zero_scale_input_dff],
                                                   axis=0)

        # INPUT DATA WITH REMOVING OUTLIERS
        x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier = dff_input_outlier.loc[:, ].values
        x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)

        # x_scale_input_outlier = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(x_scale_input_outlier)
        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min

        # x_scale_input_outlier = rescale(x_scale_input_outlier, new_min=0, new_max=1)
        pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
        principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_input_outlier))])
        finalDf_scale_input_outlier = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
        Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_outlier = pca_scale_input_outlier.components_.T * np.sqrt(
            pca_scale_input_outlier.explained_variance_)
        loading_scale_input_outlier_df = pd.DataFrame(data=loading_scale_input_outlier[:, 0:2],
                                                      columns=["PC1", "PC2"])
        line_group_scale_input_outlier_df = pd.DataFrame(data=features_input_outlier, columns=['line_group'])
        loading_scale_input_outlier_dff = pd.concat([loading_scale_input_outlier_df, line_group_scale_input_outlier_df],
                                                    axis=1)
        a = (len(features_input_outlier), 2)
        zero_scale_input_outlier = np.zeros(a)
        zero_scale_input_outlier_df = pd.DataFrame(data=zero_scale_input_outlier, columns=["PC1", "PC2"])
        zero_scale_input_outlier_dff = pd.concat([zero_scale_input_outlier_df, line_group_scale_input_outlier_df],
                                                 axis=1)
        loading_scale_input_outlier_line_graph = pd.concat(
            [loading_scale_input_outlier_dff, zero_scale_input_outlier_dff],
            axis=0)
        # COVARIANCE MATRIX
        x_scale_input_covar = dff_input.loc[:, features_input].values
        y_scale_input_covar = dff_input.loc[:, ].values
        pca_scale_input_covar = PCA(n_components=len(features_input))
        principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
        principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                     , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target], axis=1)
        dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
        Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_covar = pca_scale_input_covar.components_.T * np.sqrt(
            pca_scale_input_covar.explained_variance_)
        loading_scale_input_df_covar = pd.DataFrame(data=loading_scale_input_covar[:, 0:2],
                                                    columns=["PC1", "PC2"])
        line_group_scale_input_df_covar = pd.DataFrame(data=features_input, columns=['line_group'])
        loading_scale_input_dff_covar = pd.concat([loading_scale_input_df_covar, line_group_scale_input_df_covar],
                                                  axis=1)
        a = (len(features_input), 2)
        zero_scale_input_covar = np.zeros(a)
        zero_scale_input_df_covar = pd.DataFrame(data=zero_scale_input_covar, columns=["PC1", "PC2"])
        zero_scale_input_dff_covar = pd.concat([zero_scale_input_df_covar, line_group_scale_input_df_covar], axis=1)
        loading_scale_input_line_graph_covar = pd.concat([loading_scale_input_dff_covar, zero_scale_input_dff_covar],
                                                         axis=0)
        # COVARIANCE MATRIX OUTLIERS
        x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
        pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
            x_scale_input_outlier_covar)
        principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                             , columns=['PC' + str(i + 1) for i in
                                                                        range(len(features_input_outlier))])
        finalDf_scale_input_outlier_covar = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
        Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_outlier_covar = pca_scale_input_outlier_covar.components_.T * np.sqrt(
            pca_scale_input_outlier_covar.explained_variance_)
        loading_scale_input_outlier_df_covar = pd.DataFrame(data=loading_scale_input_outlier_covar[:, 0:2],
                                                            columns=["PC1", "PC2"])
        line_group_scale_input_outlier_df_covar = pd.DataFrame(data=features_input_outlier, columns=['line_group'])
        loading_scale_input_outlier_dff_covar = pd.concat([loading_scale_input_outlier_df_covar,
                                                           line_group_scale_input_outlier_df_covar],
                                                          axis=1)
        a = (len(features_input_outlier), 2)
        zero_scale_input_outlier_covar = np.zeros(a)
        zero_scale_input_outlier_df_covar = pd.DataFrame(data=zero_scale_input_outlier_covar, columns=["PC1", "PC2"])
        zero_scale_input_outlier_dff_covar = pd.concat([zero_scale_input_outlier_df_covar,
                                                        line_group_scale_input_outlier_df_covar],
                                                       axis=1)
        loading_scale_input_outlier_line_graph_covar = pd.concat(
            [loading_scale_input_outlier_dff_covar, zero_scale_input_outlier_dff_covar],
            axis=0)
        if outlier == 'No' and matrix_type == "Correlation":
            dat = dfff_scale_input
            variance = Var_scale_input
        elif outlier == 'Yes' and matrix_type == "Correlation":
            dat = dfff_scale_input_outlier
            variance = Var_scale_input_outlier
        elif outlier == "No" and matrix_type == "Covariance":
            dat = dfff_scale_input_covar
            variance = Var_scale_input_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dat = dfff_scale_input_outlier_covar
            variance = Var_scale_input_outlier_covar
        trace2 = go.Scatter(x=dat['PC1'], y=dat['PC2'], mode='markers',
                            marker_color=dat[color] if target == 'Yes' else None,
                            marker_size=dat[size] if target2 == 'Yes' else 12,
                            text=dat[dat.columns[0]],
                            hovertemplate=
                            '<b>%{text}</b>' +
                            '<br>PC1: %{x}<br>' +
                            'PC2: %{y}'
                            "<extra></extra>",
                            marker=dict(opacity=0.7, colorscale='Plasma',
                                        sizeref=max(dat[size]) / (15 ** 2) if target2 == 'Yes' else None,
                                        sizemode='area',
                                        showscale=True if target == 'Yes' else False,
                                        line=dict(width=0.5, color='DarkSlateGrey'),
                                        colorbar=dict(title=dict(text=color if target == 'Yes' else None,
                                                                 font=dict(family='Helvetica'),
                                                                 side='right'), ypad=0),
                                        ),
                            )
        ####################################################################################################
        if outlier == 'No' and matrix_type == "Correlation":
            data = loading_scale_input_line_graph
        elif outlier == 'Yes' and matrix_type == "Correlation":
            data = loading_scale_input_outlier_line_graph
        elif outlier == "No" and matrix_type == "Covariance":
            data = loading_scale_input_line_graph_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            data = loading_scale_input_outlier_line_graph_covar
        counter = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf = data[data['line_group'] == i]
            trace1 = go.Scatter(x=dataf['PC1'], y=dataf['PC2'],
                                line=dict(color="#666666" if target == 'Yes' else '#4f4f4f'), name=i,
                                # text=i,
                                meta=i,
                                hovertemplate=
                                '<b>%{meta}</b>' +
                                '<br>PC1: %{x}<br>' +
                                'PC2: %{y}'
                                "<extra></extra>",
                                mode='lines+text', textposition='bottom right', textfont=dict(size=12),
                                )
            lists[counter] = trace1
            counter = counter + 1
        ####################################################################################################
        if graph_update == 'Biplot':
            lists.insert(0, trace2)
            return {'data': lists,
                    'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2))),
                                        yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2))),
                                        showlegend=False, margin={'r': 0},
                                        # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                        #              y0=-1, x1=1, y1=1,
                                        #              line_color="DarkSlateGrey")]
                                        ),
                    }
        elif graph_update == 'Loadings':
            return {'data': lists,
                    'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2))),
                                        yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2))),
                                        showlegend=False, margin={'r': 0},
                                        # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                        #              y0=-1, x1=1, y1=1,
                                        #              line_color="DarkSlateGrey")]
                                        ),

                    }


@app.callback(
    Output('size-second-target-container', 'children'),
    [Input('size-scale-scores', 'value'),
     Input('outlier-value-biplot', 'value'),
     Input('csv-data', 'data')
     ]
)
def update_output(size, outlier, data):
    if not data:
        return dash.no_update
    if size is None:
        raise dash.exceptions.PreventUpdate
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    z_scores_dff_size = scipy.stats.zscore(dff)
    abs_z_scores_dff_size = np.abs(z_scores_dff_size)
    filtered_entries_dff_size = (abs_z_scores_dff_size < 3).all(axis=1)
    dff_target_outlier_size = dff[filtered_entries_dff_size]
    if outlier == 'Yes':
        size_range = [round(dff_target_outlier_size[size].min(), 2), round(dff_target_outlier_size[size].max(), 2)]
    elif outlier == 'No':
        size_range = [round(dff[size].min(), 2), round(dff[size].max(), 2)]
    return '{}'.format(size_range)


@app.callback(Output('cos2-plot', 'figure'),
              [
                  Input('outlier-value-cos2', 'value'),
                  Input('feature-input', 'value'),
                  Input('all-custom-choice', 'value'),
                  Input("matrix-type-cos2", "value"),
                  Input('csv-data', 'data')
              ])
def update_cos2_plot(outlier, input, all_custom, matrix_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        # x_scale = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(x_scale)
        features1 = dff.columns
        features = list(features1)
        # OUTLIER DATA
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min

        # ORIGINAL DATA WITH OUTLIERS
        x_scale = dff.loc[:, features].values
        y_scale = dff.loc[:, ].values
        x_scale = StandardScaler().fit_transform(x_scale)
        pca_scale = PCA(n_components=len(features))
        principalComponents_scale = pca_scale.fit_transform(x_scale)
        principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                         , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale = pd.concat([df[[df.columns[0]]], principalDf_scale], axis=1)
        Var_scale = pca_scale.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale = pca_scale.components_.T * np.sqrt(pca_scale.explained_variance_)
        loading_scale_df = pd.DataFrame(data=loading_scale[:, 0:2],
                                        columns=["PC1", "PC2"])
        loading_scale_df['cos2'] = (loading_scale_df["PC1"] ** 2) + (loading_scale_df["PC2"] ** 2)
        line_group_scale_df = pd.DataFrame(data=features, columns=['line_group'])
        loading_scale_dff = pd.concat([loading_scale_df, line_group_scale_df], axis=1)
        a = (len(features), 2)
        zero_scale = np.zeros(a)
        zero_scale_df = pd.DataFrame(data=zero_scale, columns=["PC1", "PC2"])
        zero_scale_df_color = pd.DataFrame(data=loading_scale_df.iloc[:, 2], columns=['cos2'])
        zero_scale_dff = pd.concat([zero_scale_df, zero_scale_df_color, line_group_scale_df], axis=1)
        loading_scale_line_graph = pd.concat([loading_scale_dff, zero_scale_dff], axis=0)

        # ORIGINAL DATA WITH REMOVING OUTLIERS
        x_outlier_scale = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale = outlier_dff.loc[:, ].values
        x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)

        # x_outlier_scale = MinMaxScaler().fit_transform(x_outlier_scale)

        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min
        #
        # x_outlier_scale = rescale(x_outlier_scale, new_min=0, new_max=1)
        # uses covariance matrix
        pca_outlier_scale = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
        principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                 , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier_scale = pd.concat([outlier_names, principalDf_outlier_scale], axis=1)
        Var_outlier_scale = pca_outlier_scale.explained_variance_ratio_
        # calculating loading
        # calculating loading vector plot
        loading_outlier_scale = pca_outlier_scale.components_.T * np.sqrt(pca_outlier_scale.explained_variance_)
        loading_outlier_scale_df = pd.DataFrame(data=loading_outlier_scale[:, 0:2],
                                                columns=["PC1", "PC2"])
        loading_outlier_scale_df["cos2"] = (loading_outlier_scale_df["PC1"] ** 2) + (
                loading_outlier_scale_df["PC2"] ** 2)
        line_group_df = pd.DataFrame(data=features_outlier, columns=['line_group'])
        loading_outlier_scale_dff = pd.concat([loading_outlier_scale_df, line_group_df], axis=1)
        a = (len(features_outlier), 2)
        zero_outlier_scale = np.zeros(a)
        zero_outlier_scale_df = pd.DataFrame(data=zero_outlier_scale, columns=["PC1", "PC2"])
        zero_outlier_scale_df_color = pd.DataFrame(data=loading_outlier_scale_df.iloc[:, 2], columns=['cos2'])
        zero_outlier_scale_dff = pd.concat([zero_outlier_scale_df, zero_outlier_scale_df_color, line_group_df], axis=1)
        loading_outlier_scale_line_graph = pd.concat([loading_outlier_scale_dff, zero_outlier_scale_dff], axis=0)

        loading_scale_line_graph_sort = loading_scale_line_graph.sort_values(by='cos2')
        loading_outlier_scale_line_graph_sort = loading_outlier_scale_line_graph.sort_values(by='cos2')

        # COVARIANCE MATRIX
        x_scale_covar = dff.loc[:, features].values
        y_scale_covar = dff.loc[:, ].values
        pca_scale_covar = PCA(n_components=len(features))
        principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
        principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                               , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_covar], axis=1)
        Var_scale_covar = pca_scale_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_covar = pca_scale_covar.components_.T * np.sqrt(pca_scale_covar.explained_variance_)
        loading_scale_df_covar = pd.DataFrame(data=loading_scale_covar[:, 0:2],
                                              columns=["PC1", "PC2"])
        loading_scale_df_covar['cos2'] = (loading_scale_df_covar["PC1"] ** 2) + (loading_scale_df_covar["PC2"] ** 2)
        line_group_scale_df_covar = pd.DataFrame(data=features, columns=['line_group'])
        loading_scale_dff_covar = pd.concat([loading_scale_df_covar, line_group_scale_df_covar], axis=1)
        a = (len(features), 2)
        zero_scale_covar = np.zeros(a)
        zero_scale_df_covar = pd.DataFrame(data=zero_scale_covar, columns=["PC1", "PC2"])
        zero_scale_df_color_covar = pd.DataFrame(data=loading_scale_df_covar.iloc[:, 2], columns=['cos2'])
        zero_scale_dff_covar = pd.concat([zero_scale_df_covar, zero_scale_df_color_covar, line_group_scale_df_covar],
                                         axis=1)
        loading_scale_line_graph_covar = pd.concat([loading_scale_dff_covar, zero_scale_dff_covar], axis=0)
        loading_scale_line_graph_sort_covar = loading_scale_line_graph_covar.sort_values(by='cos2')

        # COVARIANCE MATRIX WITH OUTLIERS
        x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale_covar = outlier_dff.loc[:, ].values
        pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
        principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier_scale_covar = pd.concat([outlier_names, principalDf_outlier_scale_covar], axis=1)
        Var_outlier_scale_covar = pca_outlier_scale_covar.explained_variance_ratio_
        # calculating loading
        # calculating loading vector plot
        loading_outlier_scale_covar = pca_outlier_scale_covar.components_.T * np.sqrt(
            pca_outlier_scale_covar.explained_variance_)
        loading_outlier_scale_df_covar = pd.DataFrame(data=loading_outlier_scale_covar[:, 0:2],
                                                      columns=["PC1", "PC2"])
        loading_outlier_scale_df_covar["cos2"] = (loading_outlier_scale_df_covar["PC1"] ** 2) + (
                loading_outlier_scale_df_covar["PC2"] ** 2)
        line_group_df_covar = pd.DataFrame(data=features_outlier, columns=['line_group'])
        loading_outlier_scale_dff_covar = pd.concat([loading_outlier_scale_df_covar, line_group_df_covar], axis=1)
        a = (len(features_outlier), 2)
        zero_outlier_scale_covar = np.zeros(a)
        zero_outlier_scale_df_covar = pd.DataFrame(data=zero_outlier_scale_covar, columns=["PC1", "PC2"])
        zero_outlier_scale_df_color_covar = pd.DataFrame(data=loading_outlier_scale_df_covar.iloc[:, 2],
                                                         columns=['cos2'])
        zero_outlier_scale_dff_covar = pd.concat([zero_outlier_scale_df_covar, zero_outlier_scale_df_color_covar,
                                                  line_group_df_covar], axis=1)
        loading_outlier_scale_line_graph_covar = pd.concat(
            [loading_outlier_scale_dff_covar, zero_outlier_scale_dff_covar], axis=0)
        loading_outlier_scale_line_graph_sort_covar = loading_outlier_scale_line_graph_covar.sort_values(by='cos2')

        # scaling data
        if outlier == 'No' and matrix_type == "Correlation":
            data = loading_scale_line_graph_sort
            variance = Var_scale
        elif outlier == 'Yes' and matrix_type == "Correlation":
            data = loading_outlier_scale_line_graph_sort
            variance = Var_outlier_scale
        elif outlier == "No" and matrix_type == "Covariance":
            data = loading_scale_line_graph_sort_covar
            variance = Var_scale_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            data = loading_outlier_scale_line_graph_sort_covar
            variance = Var_outlier_scale_covar
        N = len(data['cos2'].unique())
        end_color = "#00264c"  # dark blue
        start_color = "#c6def5"  # light blue
        colorscale = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]
        counter = 0
        counter_color = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf_all = data[data['line_group'] == i]
            trace1_all = go.Scatter(x=dataf_all['PC1'], y=dataf_all['PC2'], mode='lines+text',
                                    name=i, line=dict(color=colorscale[counter_color]),
                                    # text=i,
                                    meta=i,
                                    hovertemplate=
                                    '<b>%{meta}</b>' +
                                    '<br>PC1: %{x}<br>' +
                                    'PC2: %{y}'
                                    "<extra></extra>",
                                    textposition='bottom right', textfont=dict(size=12)
                                    )
            trace2_all = go.Scatter(x=[1, -1], y=[1, -1], mode='markers',
                                    hoverinfo='skip',
                                    marker=dict(showscale=True, opacity=0,
                                                color=[data["cos2"].min(), data["cos2"].max()],
                                                colorscale=colorscale,
                                                colorbar=dict(title=dict(text="Cos2",
                                                                         side='right'), ypad=0)
                                                ), )
            lists[counter] = trace1_all
            counter = counter + 1
            counter_color = counter_color + 1
            lists.append(trace2_all)
        ####################################################################################################
        return {'data': lists,
                'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2)), mirror=True,
                                               ticks='outside', showline=True),
                                    yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2)), mirror=True,
                                               ticks='outside', showline=True),
                                    showlegend=False, margin={'r': 0},
                                    # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                    #              y0=-1, x1=1, y1=1,
                                    #              line_color="DarkSlateGrey")]
                                    ),

                }
    elif all_custom == "Custom":
        # Dropping Data variables
        dff_input = dff.drop(columns=dff[input])
        features1_input = dff_input.columns
        features_input = list(features1_input)
        dff_target = dff[input]
        # OUTLIER DATA INPUT
        z_scores_input = scipy.stats.zscore(dff_input)
        abs_z_scores_input = np.abs(z_scores_input)
        filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
        dff_input_outlier = dff_input[filtered_entries_input]
        features1_input_outlier = dff_input_outlier.columns
        features_input_outlier = list(features1_input_outlier)
        outlier_names_input1 = df[filtered_entries_input]
        outlier_names_input = outlier_names_input1.iloc[:, 0]
        # OUTLIER DATA TARGET
        z_scores_target = scipy.stats.zscore(dff_target)
        abs_z_scores_target = np.abs(z_scores_target)
        filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
        dff_target_outlier = dff_target[filtered_entries_target]
        # INPUT DATA WITH OUTLIERS
        x_scale_input = dff_input.loc[:, features_input].values
        y_scale_input = dff_input.loc[:, ].values
        x_scale_input = StandardScaler().fit_transform(x_scale_input)

        # # x_scale_input = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(x_scale_input)
        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min
        #
        # x_scale_input = rescale(x_scale_input, new_min=0, new_max=1)

        pca_scale_input = PCA(n_components=len(features_input))
        principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
        principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                               , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
        dfff_scale_input = finalDf_scale_input.fillna(0)
        Var_scale_input = pca_scale_input.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input = pca_scale_input.components_.T * np.sqrt(pca_scale_input.explained_variance_)
        loading_scale_input_df = pd.DataFrame(data=loading_scale_input[:, 0:2],
                                              columns=["PC1", "PC2"])
        loading_scale_input_df["cos2"] = (loading_scale_input_df["PC1"] ** 2) + (loading_scale_input_df["PC2"] ** 2)
        line_group_scale_input_df = pd.DataFrame(data=features_input, columns=['line_group'])
        loading_scale_input_dff = pd.concat([loading_scale_input_df, line_group_scale_input_df],
                                            axis=1)
        a = (len(features_input), 2)
        zero_scale_input = np.zeros(a)
        zero_scale_input_df = pd.DataFrame(data=zero_scale_input, columns=["PC1", "PC2"])
        zero_scale_input_df_color = pd.DataFrame(data=loading_scale_input_df.iloc[:, 2], columns=['cos2'])
        zero_scale_input_dff = pd.concat([zero_scale_input_df, zero_scale_input_df_color, line_group_scale_input_df],
                                         axis=1)
        loading_scale_input_line_graph = pd.concat([loading_scale_input_dff, zero_scale_input_dff],
                                                   axis=0)
        # INPUT DATA WITH REMOVING OUTLIERS
        x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier = dff_input_outlier.loc[:, ].values
        x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)

        # # x_scale_input_outlier = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(x_scale_input_outlier)
        # def rescale(data, new_min=0, new_max=1):
        #     """Rescale the data to be within the range [new_min, new_max]"""
        #     return (data - data.min()) / (data.max() - data.min()) * (new_max - new_min) + new_min

        # x_scale_input_outlier = rescale(x_scale_input_outlier, new_min=0, new_max=1)
        pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
        principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_input_outlier))])
        finalDf_scale_input_outlier = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
        Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_outlier = pca_scale_input_outlier.components_.T * np.sqrt(
            pca_scale_input_outlier.explained_variance_)
        loading_scale_input_outlier_df = pd.DataFrame(data=loading_scale_input_outlier[:, 0:2],
                                                      columns=["PC1", "PC2"])
        loading_scale_input_outlier_df["cos2"] = (loading_scale_input_outlier_df["PC1"] ** 2) + \
                                                 (loading_scale_input_outlier_df["PC2"] ** 2)
        line_group_scale_input_outlier_df = pd.DataFrame(data=features_input_outlier, columns=['line_group'])
        loading_scale_input_outlier_dff = pd.concat([loading_scale_input_outlier_df, line_group_scale_input_outlier_df],
                                                    axis=1)
        a = (len(features_input_outlier), 2)
        zero_scale_input_outlier = np.zeros(a)
        zero_scale_input_outlier_df = pd.DataFrame(data=zero_scale_input_outlier, columns=["PC1", "PC2"])
        zero_scale_input_outlier_df_color = pd.DataFrame(data=loading_scale_input_outlier_df.iloc[:, 2],
                                                         columns=['cos2'])
        zero_scale_input_outlier_dff = pd.concat([zero_scale_input_outlier_df, zero_scale_input_outlier_df_color,
                                                  line_group_scale_input_outlier_df],
                                                 axis=1)
        loading_scale_input_outlier_line_graph = pd.concat(
            [loading_scale_input_outlier_dff, zero_scale_input_outlier_dff],
            axis=0)
        loading_scale_input_line_graph_sort = loading_scale_input_line_graph.sort_values(by='cos2')
        loading_scale_input_outlier_line_graph_sort = loading_scale_input_outlier_line_graph.sort_values(by='cos2')

        # COVARIANCE MATRIX
        x_scale_input_covar = dff_input.loc[:, features_input].values
        y_scale_input_covar = dff_input.loc[:, ].values
        pca_scale_input_covar = PCA(n_components=len(features_input))
        principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
        principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                     , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target], axis=1)
        dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
        Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_covar = pca_scale_input_covar.components_.T * np.sqrt(
            pca_scale_input_covar.explained_variance_)
        loading_scale_input_df_covar = pd.DataFrame(data=loading_scale_input_covar[:, 0:2],
                                                    columns=["PC1", "PC2"])
        loading_scale_input_df_covar["cos2"] = (loading_scale_input_df_covar["PC1"] ** 2) + (
                loading_scale_input_df_covar["PC2"] ** 2)
        line_group_scale_input_df_covar = pd.DataFrame(data=features_input, columns=['line_group'])
        loading_scale_input_dff_covar = pd.concat([loading_scale_input_df_covar, line_group_scale_input_df_covar],
                                                  axis=1)
        a = (len(features_input), 2)
        zero_scale_input_covar = np.zeros(a)
        zero_scale_input_df_covar = pd.DataFrame(data=zero_scale_input_covar, columns=["PC1", "PC2"])
        zero_scale_input_df_color_covar = pd.DataFrame(data=loading_scale_input_df_covar.iloc[:, 2], columns=['cos2'])
        zero_scale_input_dff_covar = pd.concat([zero_scale_input_df_covar, zero_scale_input_df_color_covar,
                                                line_group_scale_input_df_covar],
                                               axis=1)
        loading_scale_input_line_graph_covar = pd.concat([loading_scale_input_dff_covar, zero_scale_input_dff_covar],
                                                         axis=0)
        loading_scale_input_line_graph_sort_covar = loading_scale_input_line_graph_covar.sort_values(by='cos2')

        # COVARIANCE MATRIX WITH OUTLIERS
        x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
        pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
            x_scale_input_outlier_covar)
        principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                             , columns=['PC' + str(i + 1) for i in
                                                                        range(len(features_input_outlier))])
        finalDf_scale_input_outlier_covar = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
        Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_outlier_covar = pca_scale_input_outlier_covar.components_.T * np.sqrt(
            pca_scale_input_outlier_covar.explained_variance_)
        loading_scale_input_outlier_df_covar = pd.DataFrame(data=loading_scale_input_outlier_covar[:, 0:2],
                                                            columns=["PC1", "PC2"])
        loading_scale_input_outlier_df_covar["cos2"] = (loading_scale_input_outlier_df_covar["PC1"] ** 2) + \
                                                       (loading_scale_input_outlier_df_covar["PC2"] ** 2)
        line_group_scale_input_outlier_df_covar = pd.DataFrame(data=features_input_outlier, columns=['line_group'])
        loading_scale_input_outlier_dff_covar = pd.concat([loading_scale_input_outlier_df_covar,
                                                           line_group_scale_input_outlier_df_covar],
                                                          axis=1)
        a = (len(features_input_outlier), 2)
        zero_scale_input_outlier_covar = np.zeros(a)
        zero_scale_input_outlier_df_covar = pd.DataFrame(data=zero_scale_input_outlier_covar, columns=["PC1", "PC2"])
        zero_scale_input_outlier_df_color_covar = pd.DataFrame(data=loading_scale_input_outlier_df_covar.iloc[:, 2],
                                                               columns=['cos2'])
        zero_scale_input_outlier_dff_covar = pd.concat([zero_scale_input_outlier_df_covar,
                                                        zero_scale_input_outlier_df_color_covar,
                                                        line_group_scale_input_outlier_df_covar],
                                                       axis=1)
        loading_scale_input_outlier_line_graph_covar = pd.concat(
            [loading_scale_input_outlier_dff_covar, zero_scale_input_outlier_dff_covar],
            axis=0)
        loading_scale_input_outlier_line_graph_sort_covar = loading_scale_input_outlier_line_graph_covar.sort_values(
            by='cos2')
        ####################################################################################################
        if outlier == 'No' and matrix_type == "Correlation":
            data = loading_scale_input_line_graph_sort
            variance = Var_scale_input
        elif outlier == 'Yes' and matrix_type == "Correlation":
            variance = Var_scale_input_outlier
            data = loading_scale_input_outlier_line_graph_sort
        elif outlier == "No" and matrix_type == "Covariance":
            data = loading_scale_input_line_graph_sort_covar
            variance = Var_scale_input_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            variance = Var_scale_input_outlier_covar
            data = loading_scale_input_outlier_line_graph_sort_covar
        N = len(data['cos2'].unique())
        end_color = "#00264c"  # dark blue
        start_color = "#c6def5"  # light blue
        colorscale = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]
        counter_color = 0
        counter = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf = data[data['line_group'] == i]
            trace1 = go.Scatter(x=dataf['PC1'], y=dataf['PC2'], name=i, line=dict(color=colorscale[counter_color]),
                                mode='lines+text', textposition='bottom right', textfont=dict(size=12),
                                meta=i,
                                hovertemplate=
                                '<b>%{meta}</b>' +
                                '<br>PC1: %{x}<br>' +
                                'PC2: %{y}'
                                "<extra></extra>",
                                )
            trace2_all = go.Scatter(x=[1, -1], y=[1, -1], mode='markers', hoverinfo='skip',
                                    marker=dict(showscale=True, color=[data["cos2"].min(), data["cos2"].max()],
                                                colorscale=colorscale, opacity=0,
                                                colorbar=dict(title=dict(text="Cos2",
                                                                         side='right'), ypad=0)
                                                ), )
            lists[counter] = trace1
            counter_color = counter_color + 1
            counter = counter + 1
            lists.append(trace2_all)
        ####################################################################################################
        return {'data': lists,
                'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2)),
                                               mirror=True, ticks='outside', showline=True),
                                    yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2)),
                                               mirror=True, ticks='outside', showline=True),
                                    showlegend=False, margin={'r': 0},
                                    # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                    #              y0=-1, x1=1, y1=1,
                                    #              line_color="DarkSlateGrey")]
                                    ),

                }


@app.callback(Output('contrib-plot', 'figure'),
              [
                  Input('outlier-value-contrib', 'value'),
                  Input('feature-input', 'value'),
                  Input('all-custom-choice', 'value'),
                  Input("matrix-type-contrib", "value"),
                  Input('csv-data', 'data')
              ])
def update_cos2_plot(outlier, input, all_custom, matrix_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        features1 = dff.columns
        features = list(features1)
        # OUTLIER DATA
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        x_scale = dff.loc[:, features].values
        y_scale = dff.loc[:, ].values
        x_scale = StandardScaler().fit_transform(x_scale)
        pca_scale = PCA(n_components=len(features))
        principalComponents_scale = pca_scale.fit_transform(x_scale)
        principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                         , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale = pd.concat([df[[df.columns[0]]], principalDf_scale], axis=1)
        Var_scale = pca_scale.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale = pca_scale.components_.T * np.sqrt(pca_scale.explained_variance_)
        loading_scale_df = pd.DataFrame(data=loading_scale[:, 0:2],
                                        columns=["PC1", "PC2"])
        loading_scale_df["PC1_cos2"] = loading_scale_df["PC1"] ** 2
        loading_scale_df["PC2_cos2"] = loading_scale_df["PC2"] ** 2
        loading_scale_df["PC1_contrib"] = \
            (loading_scale_df["PC1_cos2"] * 100) / (loading_scale_df["PC1_cos2"].sum(axis=0))
        loading_scale_df["PC2_contrib"] = \
            (loading_scale_df["PC2_cos2"] * 100) / (loading_scale_df["PC2_cos2"].sum(axis=0))
        loading_scale_df["contrib"] = loading_scale_df["PC1_contrib"] + loading_scale_df["PC2_contrib"]
        # after youve got sum of contrib (colorscale) get that and PC1 and PC2 into a sep df
        loading_scale_dataf = pd.concat([loading_scale_df.iloc[:, 0:2], loading_scale_df.iloc[:, 6]], axis=1)
        line_group_scale_df = pd.DataFrame(data=features, columns=['line_group'])
        loading_scale_dff = pd.concat([loading_scale_dataf, line_group_scale_df], axis=1)
        a = (len(features), 2)
        zero_scale = np.zeros(a)
        zero_scale_df = pd.DataFrame(data=zero_scale, columns=["PC1", "PC2"])
        zero_scale_df_color = pd.DataFrame(data=loading_scale_dataf.iloc[:, 2], columns=['contrib'])
        zero_scale_dff = pd.concat([zero_scale_df, zero_scale_df_color, line_group_scale_df], axis=1)
        loading_scale_line_graph = pd.concat([loading_scale_dff, zero_scale_dff], axis=0)

        # ORIGINAL DATA WITH REMOVING OUTLIERS
        x_outlier_scale = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale = outlier_dff.loc[:, ].values
        x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)

        pca_outlier_scale = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
        principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                 , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        finalDf_outlier_scale = pd.concat([outlier_names, principalDf_outlier_scale], axis=1)
        Var_outlier_scale = pca_outlier_scale.explained_variance_ratio_

        loading_outlier_scale = pca_outlier_scale.components_.T * np.sqrt(pca_outlier_scale.explained_variance_)
        loading_outlier_scale_df = pd.DataFrame(data=loading_outlier_scale[:, 0:2],
                                                columns=["PC1", "PC2"])
        loading_outlier_scale_df["PC1_cos2"] = loading_outlier_scale_df["PC1"] ** 2
        loading_outlier_scale_df["PC2_cos2"] = loading_outlier_scale_df["PC2"] ** 2
        loading_outlier_scale_df["PC1_contrib"] = \
            (loading_outlier_scale_df["PC1_cos2"] * 100) / (loading_outlier_scale_df["PC1_cos2"].sum(axis=0))
        loading_outlier_scale_df["PC2_contrib"] = \
            (loading_outlier_scale_df["PC2_cos2"] * 100) / (loading_outlier_scale_df["PC2_cos2"].sum(axis=0))
        loading_outlier_scale_df["contrib"] = loading_outlier_scale_df["PC1_contrib"] + loading_outlier_scale_df[
            "PC2_contrib"]
        # after youve got sum of contrib (colorscale) get that and PC1 and PC2 into a sep df
        loading_outlier_scale_dataf = pd.concat(
            [loading_outlier_scale_df.iloc[:, 0:2], loading_outlier_scale_df.iloc[:, 6]], axis=1)

        line_group_df = pd.DataFrame(data=features_outlier, columns=['line_group'])
        loading_outlier_scale_dff = pd.concat([loading_outlier_scale_dataf, line_group_df], axis=1)
        a = (len(features_outlier), 2)
        zero_outlier_scale = np.zeros(a)
        zero_outlier_scale_df = pd.DataFrame(data=zero_outlier_scale, columns=["PC1", "PC2"])
        zero_outlier_scale_df_color = pd.DataFrame(data=loading_outlier_scale_dataf.iloc[:, 2], columns=['contrib'])
        zero_outlier_scale_dff = pd.concat([zero_outlier_scale_df, zero_outlier_scale_df_color, line_group_df], axis=1)
        loading_outlier_scale_line_graph = pd.concat([loading_outlier_scale_dff, zero_outlier_scale_dff], axis=0)

        loading_scale_line_graph_sort = loading_scale_line_graph.sort_values(by='contrib')
        loading_outlier_scale_line_graph_sort = loading_outlier_scale_line_graph.sort_values(by='contrib')
        # COVARIANCE MATRIX
        x_scale_covar = dff.loc[:, features].values
        y_scale_covar = dff.loc[:, ].values
        pca_scale_covar = PCA(n_components=len(features))
        principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
        principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                               , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_covar], axis=1)
        Var_scale_covar = pca_scale_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_covar = pca_scale_covar.components_.T * np.sqrt(pca_scale_covar.explained_variance_)
        loading_scale_df_covar = pd.DataFrame(data=loading_scale_covar[:, 0:2],
                                              columns=["PC1", "PC2"])
        loading_scale_df_covar["PC1_cos2"] = loading_scale_df_covar["PC1"] ** 2
        loading_scale_df_covar["PC2_cos2"] = loading_scale_df_covar["PC2"] ** 2
        loading_scale_df_covar["PC1_contrib"] = \
            (loading_scale_df_covar["PC1_cos2"] * 100) / (loading_scale_df_covar["PC1_cos2"].sum(axis=0))
        loading_scale_df_covar["PC2_contrib"] = \
            (loading_scale_df_covar["PC2_cos2"] * 100) / (loading_scale_df_covar["PC2_cos2"].sum(axis=0))
        loading_scale_df_covar["contrib"] = loading_scale_df_covar["PC1_contrib"] + loading_scale_df_covar[
            "PC2_contrib"]
        loading_scale_dataf_covar = pd.concat([loading_scale_df_covar.iloc[:, 0:2], loading_scale_df_covar.iloc[:, 6]],
                                              axis=1)
        line_group_scale_df_covar = pd.DataFrame(data=features, columns=['line_group'])
        loading_scale_dff_covar = pd.concat([loading_scale_dataf_covar, line_group_scale_df_covar], axis=1)
        a = (len(features), 2)
        zero_scale_covar = np.zeros(a)
        zero_scale_df_covar = pd.DataFrame(data=zero_scale_covar, columns=["PC1", "PC2"])
        zero_scale_df_color_covar = pd.DataFrame(data=loading_scale_dataf_covar.iloc[:, 2], columns=['contrib'])
        zero_scale_dff_covar = pd.concat([zero_scale_df_covar, zero_scale_df_color_covar, line_group_scale_df_covar],
                                         axis=1)
        loading_scale_line_graph_covar = pd.concat([loading_scale_dff_covar, zero_scale_dff_covar], axis=0)
        loading_scale_line_graph_sort_covar = loading_scale_line_graph_covar.sort_values(by='contrib')

        # COVARIANCE MATRIX OUTLIERS REMOVED
        x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale_covar = outlier_dff.loc[:, ].values
        pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
        principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_outlier))])
        finalDf_outlier_scale_covar = pd.concat([outlier_names, principalDf_outlier_scale_covar], axis=1)
        Var_outlier_scale_covar = pca_outlier_scale_covar.explained_variance_ratio_

        loading_outlier_scale_covar = pca_outlier_scale_covar.components_.T * np.sqrt(
            pca_outlier_scale_covar.explained_variance_)
        loading_outlier_scale_df_covar = pd.DataFrame(data=loading_outlier_scale_covar[:, 0:2],
                                                      columns=["PC1", "PC2"])
        loading_outlier_scale_df_covar["PC1_cos2"] = loading_outlier_scale_df_covar["PC1"] ** 2
        loading_outlier_scale_df_covar["PC2_cos2"] = loading_outlier_scale_df_covar["PC2"] ** 2
        loading_outlier_scale_df_covar["PC1_contrib"] = \
            (loading_outlier_scale_df_covar["PC1_cos2"] * 100) / (
                loading_outlier_scale_df_covar["PC1_cos2"].sum(axis=0))
        loading_outlier_scale_df_covar["PC2_contrib"] = \
            (loading_outlier_scale_df_covar["PC2_cos2"] * 100) / (
                loading_outlier_scale_df_covar["PC2_cos2"].sum(axis=0))
        loading_outlier_scale_df_covar["contrib"] = loading_outlier_scale_df_covar["PC1_contrib"] + \
                                                    loading_outlier_scale_df_covar[
                                                        "PC2_contrib"]
        # after youve got sum of contrib (colorscale) get that and PC1 and PC2 into a sep df
        loading_outlier_scale_dataf_covar = pd.concat(
            [loading_outlier_scale_df_covar.iloc[:, 0:2], loading_outlier_scale_df_covar.iloc[:, 6]], axis=1)
        line_group_df_covar = pd.DataFrame(data=features_outlier, columns=['line_group'])
        loading_outlier_scale_dff_covar = pd.concat([loading_outlier_scale_dataf_covar, line_group_df_covar], axis=1)
        a = (len(features_outlier), 2)
        zero_outlier_scale_covar = np.zeros(a)
        zero_outlier_scale_df_covar = pd.DataFrame(data=zero_outlier_scale_covar, columns=["PC1", "PC2"])
        zero_outlier_scale_df_color_covar = pd.DataFrame(data=loading_outlier_scale_dataf_covar.iloc[:, 2],
                                                         columns=['contrib'])
        zero_outlier_scale_dff_covar = pd.concat(
            [zero_outlier_scale_df_covar, zero_outlier_scale_df_color_covar, line_group_df_covar], axis=1)
        loading_outlier_scale_line_graph_covar = pd.concat(
            [loading_outlier_scale_dff_covar, zero_outlier_scale_dff_covar], axis=0)
        loading_outlier_scale_line_graph_sort_covar = loading_outlier_scale_line_graph_covar.sort_values(by='contrib')
        # scaling data
        if outlier == 'No' and matrix_type == "Correlation":
            data = loading_scale_line_graph_sort
            variance = Var_scale
        elif outlier == 'Yes' and matrix_type == "Correlation":
            data = loading_outlier_scale_line_graph_sort
            variance = Var_outlier_scale
        elif outlier == "No" and matrix_type == "Covariance":
            data = loading_scale_line_graph_sort_covar
            variance = Var_scale_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            data = loading_outlier_scale_line_graph_sort_covar
            variance = Var_outlier_scale_covar
        N = len(data['contrib'].unique())
        end_color = "#00264c"  # dark blue
        start_color = "#c6def5"  # light blue
        colorscale = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]
        counter = 0
        counter_color = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf_all = data[data['line_group'] == i]
            trace1_all = go.Scatter(x=dataf_all['PC1'], y=dataf_all['PC2'], mode='lines+text',
                                    name=i, line=dict(color=colorscale[counter_color]),
                                    textposition='bottom right', textfont=dict(size=12),
                                    meta=i,
                                    hovertemplate=
                                    '<b>%{meta}</b>' +
                                    '<br>PC1: %{x}<br>' +
                                    'PC2: %{y}'
                                    "<extra></extra>",
                                    )
            trace2_all = go.Scatter(x=[1, -1], y=[1, -1], mode='markers', hoverinfo='skip',
                                    marker=dict(showscale=True, opacity=0,
                                                color=[data["contrib"].min(), data["contrib"].max()],
                                                colorscale=colorscale,
                                                colorbar=dict(title=dict(text="Contribution",
                                                                         side='right'), ypad=0),
                                                ), )
            lists[counter] = trace1_all
            counter = counter + 1
            counter_color = counter_color + 1
            lists.append(trace2_all)
        ####################################################################################################
        return {'data': lists,
                'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2)),
                                               mirror=True, ticks='outside', showline=True),
                                    yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2)),
                                               mirror=True, ticks='outside', showline=True),
                                    showlegend=False, margin={'r': 0},
                                    # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                    #              y0=-1, x1=1, y1=1,
                                    #              line_color="DarkSlateGrey")]
                                    ),

                }
    elif all_custom == "Custom":
        # Dropping Data variables
        dff_input = dff.drop(columns=dff[input])
        features1_input = dff_input.columns
        features_input = list(features1_input)
        dff_target = dff[input]
        # OUTLIER DATA INPUT
        z_scores_input = scipy.stats.zscore(dff_input)
        abs_z_scores_input = np.abs(z_scores_input)
        filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
        dff_input_outlier = dff_input[filtered_entries_input]
        features1_input_outlier = dff_input_outlier.columns
        features_input_outlier = list(features1_input_outlier)
        outlier_names_input1 = df[filtered_entries_input]
        outlier_names_input = outlier_names_input1.iloc[:, 0]
        # OUTLIER DATA TARGET
        z_scores_target = scipy.stats.zscore(dff_target)
        abs_z_scores_target = np.abs(z_scores_target)
        filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
        dff_target_outlier = dff_target[filtered_entries_target]
        # INPUT DATA WITH OUTLIERS
        x_scale_input = dff_input.loc[:, features_input].values
        y_scale_input = dff_input.loc[:, ].values
        x_scale_input = StandardScaler().fit_transform(x_scale_input)

        pca_scale_input = PCA(n_components=len(features_input))
        principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
        principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                               , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
        dfff_scale_input = finalDf_scale_input.fillna(0)
        Var_scale_input = pca_scale_input.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input = pca_scale_input.components_.T * np.sqrt(pca_scale_input.explained_variance_)
        loading_scale_input_df = pd.DataFrame(data=loading_scale_input[:, 0:2],
                                              columns=["PC1", "PC2"])
        loading_scale_input_df["PC1_cos2"] = loading_scale_input_df["PC1"] ** 2
        loading_scale_input_df["PC2_cos2"] = loading_scale_input_df["PC2"] ** 2
        loading_scale_input_df["PC1_contrib"] = \
            (loading_scale_input_df["PC1_cos2"] * 100) / (loading_scale_input_df["PC1_cos2"].sum(axis=0))
        loading_scale_input_df["PC2_contrib"] = \
            (loading_scale_input_df["PC2_cos2"] * 100) / (loading_scale_input_df["PC2_cos2"].sum(axis=0))
        loading_scale_input_df["contrib"] = loading_scale_input_df["PC1_contrib"] + loading_scale_input_df[
            "PC2_contrib"]
        loading_scale_input_dataf = pd.concat(
            [loading_scale_input_df.iloc[:, 0:2], loading_scale_input_df.iloc[:, 6]], axis=1)

        line_group_scale_input_df = pd.DataFrame(data=features_input, columns=['line_group'])
        loading_scale_input_dff = pd.concat([loading_scale_input_dataf, line_group_scale_input_df],
                                            axis=1)
        a = (len(features_input), 2)
        zero_scale_input = np.zeros(a)
        zero_scale_input_df = pd.DataFrame(data=zero_scale_input, columns=["PC1", "PC2"])
        zero_scale_input_df_color = pd.DataFrame(data=loading_scale_input_dataf.iloc[:, 2], columns=['contrib'])
        zero_scale_input_dff = pd.concat([zero_scale_input_df, zero_scale_input_df_color, line_group_scale_input_df],
                                         axis=1)
        loading_scale_input_line_graph = pd.concat([loading_scale_input_dff, zero_scale_input_dff],
                                                   axis=0)
        # INPUT DATA WITH REMOVING OUTLIERS
        x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier = dff_input_outlier.loc[:, ].values
        x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)

        pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
        principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_input_outlier))])
        finalDf_scale_input_outlier = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
        Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_outlier = pca_scale_input_outlier.components_.T * np.sqrt(
            pca_scale_input_outlier.explained_variance_)
        loading_scale_input_outlier_df = pd.DataFrame(data=loading_scale_input_outlier[:, 0:2],
                                                      columns=["PC1", "PC2"])
        loading_scale_input_outlier_df["PC1_cos2"] = loading_scale_input_outlier_df["PC1"] ** 2
        loading_scale_input_outlier_df["PC2_cos2"] = loading_scale_input_outlier_df["PC2"] ** 2
        loading_scale_input_outlier_df["PC1_contrib"] = \
            (loading_scale_input_outlier_df["PC1_cos2"] * 100) / (
                loading_scale_input_outlier_df["PC1_cos2"].sum(axis=0))
        loading_scale_input_outlier_df["PC2_contrib"] = \
            (loading_scale_input_outlier_df["PC2_cos2"] * 100) / (
                loading_scale_input_outlier_df["PC2_cos2"].sum(axis=0))
        loading_scale_input_outlier_df["contrib"] = loading_scale_input_outlier_df["PC1_contrib"] + \
                                                    loading_scale_input_outlier_df[
                                                        "PC2_contrib"]
        loading_scale_input_outlier_dataf = pd.concat(
            [loading_scale_input_outlier_df.iloc[:, 0:2], loading_scale_input_outlier_df.iloc[:, 6]], axis=1)
        line_group_scale_input_outlier_df = pd.DataFrame(data=features_input_outlier, columns=['line_group'])
        loading_scale_input_outlier_dff = pd.concat(
            [loading_scale_input_outlier_dataf, line_group_scale_input_outlier_df],
            axis=1)
        a = (len(features_input_outlier), 2)
        zero_scale_input_outlier = np.zeros(a)
        zero_scale_input_outlier_df = pd.DataFrame(data=zero_scale_input_outlier, columns=["PC1", "PC2"])
        zero_scale_input_outlier_df_color = pd.DataFrame(data=loading_scale_input_outlier_dataf.iloc[:, 2],
                                                         columns=['contrib'])
        zero_scale_input_outlier_dff = pd.concat([zero_scale_input_outlier_df, zero_scale_input_outlier_df_color,
                                                  line_group_scale_input_outlier_df],
                                                 axis=1)
        loading_scale_input_outlier_line_graph = pd.concat(
            [loading_scale_input_outlier_dff, zero_scale_input_outlier_dff],
            axis=0)
        loading_scale_input_line_graph_sort = loading_scale_input_line_graph.sort_values(by='contrib')
        loading_scale_input_outlier_line_graph_sort = loading_scale_input_outlier_line_graph.sort_values(by='contrib')
        # COVARIANCE MATRIX
        x_scale_input_covar = dff_input.loc[:, features_input].values
        y_scale_input_covar = dff_input.loc[:, ].values
        pca_scale_input_covar = PCA(n_components=len(features_input))
        principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
        principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                     , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target], axis=1)
        dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
        Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_covar = pca_scale_input_covar.components_.T * np.sqrt(
            pca_scale_input_covar.explained_variance_)
        loading_scale_input_df_covar = pd.DataFrame(data=loading_scale_input_covar[:, 0:2],
                                                    columns=["PC1", "PC2"])
        loading_scale_input_df_covar["PC1_cos2"] = loading_scale_input_df_covar["PC1"] ** 2
        loading_scale_input_df_covar["PC2_cos2"] = loading_scale_input_df_covar["PC2"] ** 2
        loading_scale_input_df_covar["PC1_contrib"] = \
            (loading_scale_input_df_covar["PC1_cos2"] * 100) / (loading_scale_input_df_covar["PC1_cos2"].sum(axis=0))
        loading_scale_input_df_covar["PC2_contrib"] = \
            (loading_scale_input_df_covar["PC2_cos2"] * 100) / (loading_scale_input_df_covar["PC2_cos2"].sum(axis=0))
        loading_scale_input_df_covar["contrib"] = loading_scale_input_df_covar["PC1_contrib"] + \
                                                  loading_scale_input_df_covar[
                                                      "PC2_contrib"]
        loading_scale_input_dataf_covar = pd.concat(
            [loading_scale_input_df_covar.iloc[:, 0:2], loading_scale_input_df_covar.iloc[:, 6]], axis=1)

        line_group_scale_input_df_covar = pd.DataFrame(data=features_input, columns=['line_group'])
        loading_scale_input_dff_covar = pd.concat([loading_scale_input_dataf_covar, line_group_scale_input_df_covar],
                                                  axis=1)
        a = (len(features_input), 2)
        zero_scale_input_covar = np.zeros(a)
        zero_scale_input_df_covar = pd.DataFrame(data=zero_scale_input_covar, columns=["PC1", "PC2"])
        zero_scale_input_df_color_covar = pd.DataFrame(data=loading_scale_input_dataf_covar.iloc[:, 2],
                                                       columns=['contrib'])
        zero_scale_input_dff_covar = pd.concat([zero_scale_input_df_covar, zero_scale_input_df_color_covar,
                                                line_group_scale_input_df_covar],
                                               axis=1)
        loading_scale_input_line_graph_covar = pd.concat([loading_scale_input_dff_covar, zero_scale_input_dff_covar],
                                                         axis=0)
        loading_scale_input_line_graph_sort_covar = loading_scale_input_line_graph_covar.sort_values(by='contrib')
        # COVARIANCE MATRIX WITH OUTLIERS
        x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
        pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
            x_scale_input_outlier_covar)
        principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                             , columns=['PC' + str(i + 1) for i in
                                                                        range(len(features_input_outlier))])
        finalDf_scale_input_outlier_covar = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
        Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
        # calculating loading vector plot
        loading_scale_input_outlier_covar = pca_scale_input_outlier_covar.components_.T * np.sqrt(
            pca_scale_input_outlier_covar.explained_variance_)
        loading_scale_input_outlier_df_covar = pd.DataFrame(data=loading_scale_input_outlier_covar[:, 0:2],
                                                            columns=["PC1", "PC2"])
        loading_scale_input_outlier_df_covar["PC1_cos2"] = loading_scale_input_outlier_df_covar["PC1"] ** 2
        loading_scale_input_outlier_df_covar["PC2_cos2"] = loading_scale_input_outlier_df_covar["PC2"] ** 2
        loading_scale_input_outlier_df_covar["PC1_contrib"] = \
            (loading_scale_input_outlier_df_covar["PC1_cos2"] * 100) / (
                loading_scale_input_outlier_df_covar["PC1_cos2"].sum(axis=0))
        loading_scale_input_outlier_df_covar["PC2_contrib"] = \
            (loading_scale_input_outlier_df_covar["PC2_cos2"] * 100) / (
                loading_scale_input_outlier_df_covar["PC2_cos2"].sum(axis=0))
        loading_scale_input_outlier_df_covar["contrib"] = loading_scale_input_outlier_df_covar["PC1_contrib"] + \
                                                          loading_scale_input_outlier_df_covar[
                                                              "PC2_contrib"]
        loading_scale_input_outlier_dataf_covar = pd.concat(
            [loading_scale_input_outlier_df_covar.iloc[:, 0:2], loading_scale_input_outlier_df_covar.iloc[:, 6]],
            axis=1)
        line_group_scale_input_outlier_df_covar = pd.DataFrame(data=features_input_outlier, columns=['line_group'])
        loading_scale_input_outlier_dff_covar = pd.concat(
            [loading_scale_input_outlier_dataf_covar, line_group_scale_input_outlier_df_covar],
            axis=1)
        a = (len(features_input_outlier), 2)
        zero_scale_input_outlier_covar = np.zeros(a)
        zero_scale_input_outlier_df_covar = pd.DataFrame(data=zero_scale_input_outlier_covar, columns=["PC1", "PC2"])
        zero_scale_input_outlier_df_color_covar = pd.DataFrame(data=loading_scale_input_outlier_dataf_covar.iloc[:, 2],
                                                               columns=['contrib'])
        zero_scale_input_outlier_dff_covar = pd.concat(
            [zero_scale_input_outlier_df_covar, zero_scale_input_outlier_df_color_covar,
             line_group_scale_input_outlier_df_covar],
            axis=1)
        loading_scale_input_outlier_line_graph_covar = pd.concat(
            [loading_scale_input_outlier_dff_covar, zero_scale_input_outlier_dff_covar],
            axis=0)
        loading_scale_input_outlier_line_graph_sort_covar = loading_scale_input_outlier_line_graph_covar.sort_values(
            by='contrib')
        ####################################################################################################
        if outlier == 'No' and matrix_type == "Correlation":
            data = loading_scale_input_line_graph_sort
            variance = Var_scale_input
        elif outlier == 'Yes' and matrix_type == "Correlation":
            variance = Var_scale_input_outlier
            data = loading_scale_input_outlier_line_graph_sort
        elif outlier == "No" and matrix_type == "Covariance":
            data = loading_scale_input_line_graph_sort_covar
            variance = Var_scale_input_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            data = loading_scale_input_outlier_line_graph_sort_covar
            variance = Var_scale_input_outlier_covar
        N = len(data['contrib'].unique())
        end_color = "#00264c"  # dark blue
        start_color = "#c6def5"  # light blue
        colorscale = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]
        counter_color = 0
        counter = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf = data[data['line_group'] == i]
            trace1 = go.Scatter(x=dataf['PC1'], y=dataf['PC2'], name=i, line=dict(color=colorscale[counter_color]),
                                mode='lines+text', textposition='bottom right', textfont=dict(size=12),
                                meta=i,
                                hovertemplate=
                                '<b>%{meta}</b>' +
                                '<br>PC1: %{x}<br>' +
                                'PC2: %{y}'
                                "<extra></extra>",
                                )
            trace2_all = go.Scatter(x=[1, -1], y=[1, -1], mode='markers', hoverinfo='skip',
                                    marker=dict(showscale=True, color=[data["contrib"].min(), data["contrib"].max()],
                                                colorscale=colorscale, opacity=0,
                                                colorbar=dict(title=dict(text="Contribution",
                                                                         side='right'), ypad=0)
                                                ))
            lists[counter] = trace1
            counter_color = counter_color + 1
            counter = counter + 1
            lists.append(trace2_all)
        ####################################################################################################
        return {'data': lists,
                'layout': go.Layout(xaxis=dict(title='PC1 ({}%)'.format(round((variance[0] * 100), 2)),
                                               mirror=True, ticks='outside', showline=True),
                                    yaxis=dict(title='PC2 ({}%)'.format(round((variance[1] * 100), 2)),
                                               mirror=True, ticks='outside', showline=True),
                                    showlegend=False, margin={'r': 0},
                                    # shapes=[dict(type="circle", xref="x", yref="y", x0=-1,
                                    #              y0=-1, x1=1, y1=1,
                                    #              line_color="DarkSlateGrey")]
                                    ),

                }


@app.callback(Output('download-link', 'download'),
              [Input('all-custom-choice', 'value'),
               Input('eigenA-outlier', 'value'),
               Input("matrix-type-data-table", 'value')])
def update_filename(all_custom, outlier, matrix_type):
    if all_custom == 'All' and outlier == 'Yes' and matrix_type == "Correlation":
        download = 'all_variables_correlation_matrix_outliers_removed_data.csv'
    elif all_custom == 'All' and outlier == 'Yes' and matrix_type == "Covariance":
        download = 'all_variables_covariance_matrix_outliers_removed_data.csv'
    elif all_custom == 'All' and outlier == 'No' and matrix_type == "Correlation":
        download = 'all_variables_correlation_matrix_data.csv'
    elif all_custom == 'All' and outlier == 'No' and matrix_type == "Covariance":
        download = 'all_variables_covariance_matrix_data.csv'
    elif all_custom == 'Custom' and outlier == 'Yes' and matrix_type == "Correlation":
        download = 'custom_variables_correlation_matrix_outliers_removed_data.csv'
    elif all_custom == 'Custom' and outlier == 'Yes' and matrix_type == "Covariance":
        download = 'custom_variables_covariance_matrix_outliers_removed_data.csv'
    elif all_custom == 'Custom' and outlier == 'No' and matrix_type == "Correlation":
        download = 'custom_variables_correlation_matrix_data.csv'
    elif all_custom == 'Custom' and outlier == 'No' and matrix_type == "Covariance":
        download = 'custom_variables_covariance_matrix_data.csv'
    return download


@app.callback(Output('download-link', 'href'),
              [Input('all-custom-choice', 'value'),
               Input('feature-input', 'value'),
               Input('eigenA-outlier', 'value'),
               Input("matrix-type-data-table", 'value'),
               Input('csv-data', 'data')])
def update_link(all_custom, input, outlier, matrix_type, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    features1 = dff.columns
    features = list(features1)
    if all_custom == 'All':
        # OUTLIER DATA
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]

        # ORIGINAL DATA WITH OUTLIERS
        x_scale = dff.loc[:, features].values
        y_scale = dff.loc[:, ].values
        x_scale = StandardScaler().fit_transform(x_scale)
        # x_scale = rescale(x_scale, new_min=0, new_max=1)
        pca_scale = PCA(n_components=len(features))
        principalComponents_scale = pca_scale.fit_transform(x_scale)
        principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                         , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale = pd.concat([df[[df.columns[0]]], principalDf_scale], axis=1)
        dfff_scale = finalDf_scale.fillna(0)

        # ORIGINAL DATA WITH REMOVING OUTLIERS
        x_outlier_scale = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale = outlier_dff.loc[:, ].values
        x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)
        pca_outlier_scale = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
        principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                 , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
        # combining principle components and target
        finalDf_outlier_scale = pd.concat([outlier_names, principalDf_outlier_scale], axis=1)
        dfff_outlier_scale = finalDf_outlier_scale.fillna(0)
        # COVARIANCE MATRIX
        x_scale_covar = dff.loc[:, features].values
        y_scale_covar = dff.loc[:, ].values
        pca_scale_covar = PCA(n_components=len(features))
        principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
        principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                               , columns=['PC' + str(i + 1) for i in range(len(features))])
        # combining principle components and target
        finalDf_scale_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_covar], axis=1)
        dfff_scale_covar = finalDf_scale_covar.fillna(0)
        # COVARIANCE MATRIX REMOVING OUTLIERS
        x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
        y_outlier_scale_covar = outlier_dff.loc[:, ].values
        pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
        principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
        principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_outlier))])
        finalDf_outlier_scale_covar = pd.concat([outlier_names, principalDf_outlier_scale_covar], axis=1)
        dfff_outlier_scale_covar = finalDf_outlier_scale_covar.fillna(0)
        if outlier == 'No' and matrix_type == "Correlation":
            dat = dfff_scale
        elif outlier == 'Yes' and matrix_type == "Correlation":
            dat = dfff_outlier_scale
        elif outlier == "No" and matrix_type == "Covariance":
            dat = dfff_scale_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dat = dfff_outlier_scale_covar
    elif all_custom == 'Custom':
        # Dropping Data variables
        dff_input = dff.drop(columns=dff[input])
        features1_input = dff_input.columns
        features_input = list(features1_input)
        dff_target = dff[input]
        # OUTLIER DATA INPUT
        z_scores_input = scipy.stats.zscore(dff_input)
        abs_z_scores_input = np.abs(z_scores_input)
        filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
        dff_input_outlier = dff_input[filtered_entries_input]
        features1_input_outlier = dff_input_outlier.columns
        features_input_outlier = list(features1_input_outlier)
        outlier_names_input1 = df[filtered_entries_input]
        outlier_names_input = outlier_names_input1.iloc[:, 0]
        # OUTLIER DATA TARGET
        z_scores_target = scipy.stats.zscore(dff_target)
        abs_z_scores_target = np.abs(z_scores_target)
        filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
        dff_target_outlier = dff_target[filtered_entries_target]
        # INPUT DATA WITH OUTLIERS
        x_scale_input = dff_input.loc[:, features_input].values
        y_scale_input = dff_input.loc[:, ].values
        x_scale_input = StandardScaler().fit_transform(x_scale_input)
        pca_scale_input = PCA(n_components=len(features_input))
        principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
        principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                               , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
        dfff_scale_input = finalDf_scale_input.fillna(0)

        # INPUT DATA WITH REMOVING OUTLIERS
        x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier = dff_input_outlier.loc[:, ].values
        x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)
        pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
        principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                       , columns=['PC' + str(i + 1) for i in
                                                                  range(len(features_input_outlier))])
        finalDf_scale_input_outlier = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
        # COVARIANCE MATRIX
        x_scale_input_covar = dff_input.loc[:, features_input].values
        y_scale_input_covar = dff_input.loc[:, ].values
        pca_scale_input_covar = PCA(n_components=len(features_input))
        principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
        principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                     , columns=['PC' + str(i + 1) for i in range(len(features_input))])
        finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target], axis=1)
        dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
        # COVARIANCE MATRIX OUTLIERS REMOVED
        x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
        y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
        pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
        principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
            x_scale_input_outlier_covar)
        principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                             , columns=['PC' + str(i + 1) for i in
                                                                        range(len(features_input_outlier))])
        finalDf_scale_input_outlier_covar = pd.concat(
            [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
            axis=1)
        dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
        if outlier == 'No' and matrix_type == "Correlation":
            dat = dfff_scale_input
        elif outlier == 'Yes' and matrix_type == "Correlation":
            dat = dfff_scale_input_outlier
        elif outlier == "No" and matrix_type == "Covariance":
            dat = dfff_scale_input_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dat = dfff_scale_input_outlier_covar
    csv_string = dat.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return csv_string


@app.callback(Output('download-link-correlation', 'download'),
              [Input('eigenA-outlier', 'value'),
               ])
def update_filename(outlier):
    if outlier == 'Yes':
        download = 'feature_correlation_removed_outliers_data.csv'
    elif outlier == 'No':
        download = 'feature_correlation_data.csv'
    return download


@app.callback([Output('data-table-correlation', 'data'),
               Output('data-table-correlation', 'columns'),
               Output('download-link-correlation', 'href')],
              [Input("eigenA-outlier", 'value'),
               Input('csv-data', 'data')], )
def update_output(outlier, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No':
        features1 = dff.columns
        features = list(features1)
        # correlation coefficient and coefficient of determination
        correlation_dff = dff.corr(method='pearson', )
        r2_dff_table = correlation_dff * correlation_dff
        r2_dff_table.insert(0, 'Features', features)
        data_frame = r2_dff_table
    if outlier == 'Yes':
        z_scores = scipy.stats.zscore(dff)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        outlier_dff = dff[filtered_entries]
        features1_outlier = outlier_dff.columns
        features_outlier = list(features1_outlier)
        outlier_names1 = df[filtered_entries]
        outlier_names = outlier_names1.iloc[:, 0]
        # correlation coefficient and coefficient of determination
        correlation_dff_outlier = outlier_dff.corr(method='pearson', )
        r2_dff_outlier_table = correlation_dff_outlier * correlation_dff_outlier
        r2_dff_outlier_table.insert(0, 'Features', features_outlier)
        data_frame = r2_dff_outlier_table

    data = data_frame.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame.columns]
    csv_string = data_frame.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-eigenA', 'download'),
              [Input("matrix-type-data-table", 'value'),
               Input('eigenA-outlier', 'value')])
def update_filename(matrix_type, outlier):
    if outlier == 'Yes' and matrix_type == "Correlation":
        download = 'Eigen_Analysis_correlation_matrix_removed_outliers_data.csv'
    elif outlier == 'Yes' and matrix_type == "Covariance":
        download = 'Eigen_Analysis_covariance_matrix_removed_outliers_data.csv'
    elif outlier == 'No' and matrix_type == "Correlation":
        download = 'Eigen_Analysis_correlation_matrix_data.csv'
    elif outlier == "No" and matrix_type == "Covariance":
        download = 'Eigen_Analysis_covariance_matrix_data.csv'
    return download


@app.callback([Output('data-table-eigenA', 'data'),
               Output('data-table-eigenA', 'columns'),
               Output('download-link-eigenA', 'href')],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input("matrix-type-data-table", 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, matrix_type, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        if outlier == 'No' and matrix_type == "Correlation":
            features1 = dff.columns
            features = list(features1)
            x = dff.loc[:, features].values
            # Separating out the target (if any)
            # Standardizing the features to {mean, variance} = {0, 1}
            x = StandardScaler().fit_transform(x)

            pca = PCA(n_components=len(features))
            principalComponents = pca.fit_transform(x)
            principalDf = pd.DataFrame(data=principalComponents
                                       , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            finalDf = pd.concat([df[[df.columns[0]]], principalDf], axis=1)
            dfff = finalDf
            loading = pca.components_.T * np.sqrt(pca.explained_variance_)

            loading_df = pd.DataFrame(data=loading[0:, 0:], index=features,
                                      columns=['PC' + str(i + 1) for i in range(loading.shape[1])])
            loading_dff = loading_df.T
            Var = pca.explained_variance_ratio_
            PC_df = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features))],
                                 columns=['Principal Component'])
            PC_num = [float(i + 1) for i in range(len(features))]
            Var_df = pd.DataFrame(data=Var, columns=['Cumulative Proportion of Explained Variance'])
            Var_cumsum = Var_df.cumsum()
            Var_dff = pd.concat([PC_df, (Var_cumsum * 100)], axis=1)
            PC_interp = np.interp(70, Var_dff['Cumulative Proportion of Explained Variance'], PC_num)
            PC_interp_int = math.ceil(PC_interp)
            eigenvalues = pca.explained_variance_
            Eigen_df = pd.DataFrame(data=eigenvalues, columns=['Eigenvalues'])
            Eigen_dff = pd.concat([PC_df, Eigen_df], axis=1)
            Var_dfff = pd.concat([(Var_cumsum * 100)], axis=1)
            Eigen_Analysis = pd.concat([PC_df.T, Eigen_df.T, Var_df.T, Var_dfff.T], axis=0)
            Eigen_Analysis = Eigen_Analysis.rename(columns=Eigen_Analysis.iloc[0])
            Eigen_Analysis = Eigen_Analysis.drop(Eigen_Analysis.index[0])
            Eigen_Analysis.insert(loc=0, column="Principal Components",
                                  value=["Eigenvalues", "Proportion of Explained Variance",
                                         "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis
        elif outlier == 'Yes' and matrix_type == "Correlation":
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]

            # correlation coefficient and coefficient of determination
            correlation_dff_outlier = outlier_dff.corr(method='pearson', )
            r2_dff_outlier = correlation_dff_outlier * correlation_dff_outlier

            x_outlier = outlier_dff.loc[:, features_outlier].values
            # Separating out the target (if any)
            y_outlier = outlier_dff.loc[:, ].values
            # Standardizing the features
            x_outlier = StandardScaler().fit_transform(x_outlier)

            pca_outlier = PCA(n_components=len(features_outlier))
            principalComponents_outlier = pca_outlier.fit_transform(x_outlier)
            principalDf_outlier = pd.DataFrame(data=principalComponents_outlier
                                               , columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
            # combining principle components and target
            finalDf_outlier = pd.concat([outlier_names, principalDf_outlier], axis=1)
            dfff_outlier = finalDf_outlier
            # calculating loading
            loading_outlier = pca_outlier.components_.T * np.sqrt(pca_outlier.explained_variance_)
            loading_df_outlier = pd.DataFrame(data=loading_outlier[0:, 0:], index=features_outlier,
                                              columns=['PC' + str(i + 1) for i in range(loading_outlier.shape[1])])
            loading_dff_outlier = loading_df_outlier.T

            Var_outlier = pca_outlier.explained_variance_ratio_
            PC_df_outlier = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier))],
                                         columns=['Principal Component'])
            PC_num_outlier = [float(i + 1) for i in range(len(features_outlier))]
            Var_df_outlier = pd.DataFrame(data=Var_outlier, columns=['Cumulative Proportion of Explained Variance'])
            Var_cumsum_outlier = Var_df_outlier.cumsum()
            Var_dff_outlier = pd.concat([PC_df_outlier, (Var_cumsum_outlier * 100)], axis=1)
            PC_interp_outlier = np.interp(70, Var_dff_outlier['Cumulative Proportion of Explained Variance'],
                                          PC_num_outlier)
            PC_interp_int_outlier = math.ceil(PC_interp_outlier)
            eigenvalues_outlier = pca_outlier.explained_variance_
            Eigen_df_outlier = pd.DataFrame(data=eigenvalues_outlier, columns=['Eigenvalues'])
            Eigen_dff_outlier = pd.concat([PC_df_outlier, Eigen_df_outlier], axis=1)
            Var_dfff_outlier = pd.concat([Var_cumsum_outlier * 100], axis=1)
            Eigen_Analysis_Outlier = pd.concat(
                [PC_df_outlier.T, Eigen_df_outlier.T, Var_df_outlier.T, Var_dfff_outlier.T],
                axis=0)
            Eigen_Analysis_Outlier = Eigen_Analysis_Outlier.rename(columns=Eigen_Analysis_Outlier.iloc[0])
            Eigen_Analysis_Outlier = Eigen_Analysis_Outlier.drop(Eigen_Analysis_Outlier.index[0])
            Eigen_Analysis_Outlier.insert(loc=0, column="Principal Components",
                                          value=["Eigenvalues", "Proportion of Explained Variance",
                                                 "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_Outlier
        elif outlier == "No" and matrix_type == "Covariance":
            features1 = dff.columns
            features = list(features1)
            x_covar = dff.loc[:, features].values
            pca_covar = PCA(n_components=len(features))
            principalComponents_covar = pca_covar.fit_transform(x_covar)
            principalDf_covar = pd.DataFrame(data=principalComponents_covar
                                             , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            finalDf_covar = pd.concat([df[[df.columns[0]]], principalDf_covar], axis=1)
            dfff_covar = finalDf_covar
            loading_covar = pca_covar.components_.T * np.sqrt(pca_covar.explained_variance_)

            loading_df_covar = pd.DataFrame(data=loading_covar[0:, 0:], index=features,
                                            columns=['PC' + str(i + 1) for i in range(loading_covar.shape[1])])
            loading_dff_covar = loading_df_covar.T
            Var_covar = pca_covar.explained_variance_ratio_
            PC_df_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features))],
                                       columns=['Principal Component'])
            PC_num_covar = [float(i + 1) for i in range(len(features))]
            Var_df_covar = pd.DataFrame(data=Var_covar, columns=['Cumulative Proportion of Explained Variance'])
            Var_cumsum_covar = Var_df_covar.cumsum()
            Var_dff_covar = pd.concat([PC_df_covar, (Var_cumsum_covar * 100)], axis=1)
            PC_interp_covar = np.interp(70, Var_dff_covar['Cumulative Proportion of Explained Variance'], PC_num_covar)
            PC_interp_int_covar = math.ceil(PC_interp_covar)
            eigenvalues_covar = pca_covar.explained_variance_
            Eigen_df_covar = pd.DataFrame(data=eigenvalues_covar, columns=['Eigenvalues'])
            Eigen_dff_covar = pd.concat([PC_df_covar, Eigen_df_covar], axis=1)
            Var_dfff_covar = pd.concat([(Var_cumsum_covar * 100)], axis=1)
            Eigen_Analysis_covar = pd.concat([PC_df_covar.T, Eigen_df_covar.T, Var_df_covar.T, Var_dfff_covar.T],
                                             axis=0)
            Eigen_Analysis_covar = Eigen_Analysis_covar.rename(columns=Eigen_Analysis_covar.iloc[0])
            Eigen_Analysis_covar = Eigen_Analysis_covar.drop(Eigen_Analysis_covar.index[0])
            Eigen_Analysis_covar.insert(loc=0, column="Principal Components",
                                        value=["Eigenvalues", "Proportion of Explained Variance",
                                               "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]
            x_outlier_covar = outlier_dff.loc[:, features_outlier].values
            # Separating out the target (if any)
            y_outlier_covar = outlier_dff.loc[:, ].values
            pca_outlier_covar = PCA(n_components=len(features_outlier))
            principalComponents_outlier_covar = pca_outlier_covar.fit_transform(x_outlier_covar)
            principalDf_outlier_covar = pd.DataFrame(data=principalComponents_outlier_covar
                                                     ,
                                                     columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
            # combining principle components and target
            finalDf_outlier_covar = pd.concat([outlier_names, principalDf_outlier_covar], axis=1)
            dfff_outlier_covar = finalDf_outlier_covar
            # calculating loading
            loading_outlier_covar = pca_outlier_covar.components_.T * np.sqrt(pca_outlier_covar.explained_variance_)
            loading_df_outlier_covar = pd.DataFrame(data=loading_outlier_covar[0:, 0:], index=features_outlier,
                                                    columns=['PC' + str(i + 1) for i in
                                                             range(loading_outlier_covar.shape[1])])
            loading_dff_outlier_covar = loading_df_outlier_covar.T

            Var_outlier_covar = pca_outlier_covar.explained_variance_ratio_
            PC_df_outlier_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_outlier))],
                                               columns=['Principal Component'])
            PC_num_outlier_covar = [float(i + 1) for i in range(len(features_outlier))]
            Var_df_outlier_covar = pd.DataFrame(data=Var_outlier_covar,
                                                columns=['Cumulative Proportion of Explained Variance'])
            Var_cumsum_outlier_covar = Var_df_outlier_covar.cumsum()
            Var_dff_outlier_covar = pd.concat([PC_df_outlier_covar, (Var_cumsum_outlier_covar * 100)], axis=1)
            PC_interp_outlier_covar = np.interp(70,
                                                Var_dff_outlier_covar['Cumulative Proportion of Explained Variance'],
                                                PC_num_outlier_covar)
            PC_interp_int_outlier_covar = math.ceil(PC_interp_outlier_covar)
            eigenvalues_outlier_covar = pca_outlier_covar.explained_variance_
            Eigen_df_outlier_covar = pd.DataFrame(data=eigenvalues_outlier_covar, columns=['Eigenvalues'])
            Eigen_dff_outlier_covar = pd.concat([PC_df_outlier_covar, Eigen_df_outlier_covar], axis=1)
            Var_dfff_outlier_covar = pd.concat([Var_cumsum_outlier_covar * 100], axis=1)
            Eigen_Analysis_Outlier_covar = pd.concat(
                [PC_df_outlier_covar.T, Eigen_df_outlier_covar.T, Var_df_outlier_covar.T, Var_dfff_outlier_covar.T],
                axis=0)
            Eigen_Analysis_Outlier_covar = Eigen_Analysis_Outlier_covar.rename(
                columns=Eigen_Analysis_Outlier_covar.iloc[0])
            Eigen_Analysis_Outlier_covar = Eigen_Analysis_Outlier_covar.drop(Eigen_Analysis_Outlier_covar.index[0])
            Eigen_Analysis_Outlier_covar.insert(loc=0, column="Principal Components",
                                                value=["Eigenvalues", "Proportion of Explained Variance",
                                                       "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_Outlier_covar

    elif all_custom == "Custom":
        if outlier == 'No' and matrix_type == "Correlation":
            # Dropping Data variables
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            x_scale_input = dff_input.loc[:, features_input].values
            y_scale_input = dff_input.loc[:, ].values
            x_scale_input = StandardScaler().fit_transform(x_scale_input)
            # INPUT DATA WITH OUTLIERS
            pca_scale_input = PCA(n_components=len(features_input))
            principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
            principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                                   , columns=['PC' + str(i + 1) for i in range(len(features_input))])
            finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
            dfff_scale_input = finalDf_scale_input.fillna(0)
            Var_scale_input = pca_scale_input.explained_variance_ratio_
            eigenvalues_scale_input = pca_scale_input.explained_variance_
            Eigen_df_scale_input = pd.DataFrame(data=eigenvalues_scale_input, columns=["Eigenvaues"])
            PC_df_scale_input = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_input))],
                                             columns=['Principal Component'])
            Var_df_scale_input = pd.DataFrame(data=Var_scale_input,
                                              columns=['Cumulative Proportion of Explained Ratio'])
            Var_cumsum_scale_input = Var_df_scale_input.cumsum()
            Var_dfff_scale_input = pd.concat([Var_cumsum_scale_input * 100], axis=1)
            Eigen_Analysis_scale_input = pd.concat([PC_df_scale_input.T, Eigen_df_scale_input.T,
                                                    Var_df_scale_input.T, Var_dfff_scale_input.T], axis=0)
            Eigen_Analysis_scale_input = Eigen_Analysis_scale_input.rename(columns=Eigen_Analysis_scale_input.iloc[0])
            Eigen_Analysis_scale_input = Eigen_Analysis_scale_input.drop(Eigen_Analysis_scale_input.index[0])
            Eigen_Analysis_scale_input.insert(loc=0, column="Principal Components",
                                              value=["Eigenvalues", "Proportion of Explained Variance",
                                                     "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_scale_input
        elif outlier == "Yes" and matrix_type == "Correlation":
            dff_input = dff.drop(columns=dff[input])
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            dff_target = dff[input]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier = dff_input_outlier.loc[:, ].values
            x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)
            # INPUT DATA WITH REMOVING OUTLIERS
            pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
            principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                           , columns=['PC' + str(i + 1) for i in
                                                                      range(len(features_input_outlier))])
            finalDf_scale_input_outlier = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
            Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
            eigenvalues_scale_input_outlier = pca_scale_input_outlier.explained_variance_
            Eigen_df_scale_input_outlier = pd.DataFrame(data=eigenvalues_scale_input_outlier, columns=["Eigenvaues"])
            PC_df_scale_input_outlier = pd.DataFrame(
                data=['PC' + str(i + 1) for i in range(len(features_input_outlier))],
                columns=['Principal Component'])
            Var_df_scale_input_outlier = pd.DataFrame(data=Var_scale_input_outlier,
                                                      columns=['Cumulative Proportion of Explained '
                                                               'Ratio'])
            Var_cumsum_scale_input_outlier = Var_df_scale_input_outlier.cumsum()
            Var_dfff_scale_input_outlier = pd.concat([Var_cumsum_scale_input_outlier * 100], axis=1)
            Eigen_Analysis_scale_input_outlier = pd.concat([PC_df_scale_input_outlier.T, Eigen_df_scale_input_outlier.T,
                                                            Var_df_scale_input_outlier.T,
                                                            Var_dfff_scale_input_outlier.T], axis=0)
            Eigen_Analysis_scale_input_outlier = Eigen_Analysis_scale_input_outlier.rename(
                columns=Eigen_Analysis_scale_input_outlier.iloc[0])
            Eigen_Analysis_scale_input_outlier = Eigen_Analysis_scale_input_outlier.drop(
                Eigen_Analysis_scale_input_outlier.index[0])
            Eigen_Analysis_scale_input_outlier.insert(loc=0, column="Principal Components",
                                                      value=["Eigenvalues", "Proportion of Explained Variance",
                                                             "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_scale_input_outlier
        elif outlier == "No" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            x_scale_input_covar = dff_input.loc[:, features_input].values
            # INPUT DATA WITH OUTLIERS
            pca_scale_input_covar = PCA(n_components=len(features_input))
            principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
            principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                         , columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_input))])
            finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target],
                                                  axis=1)
            dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
            Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
            eigenvalues_scale_input_covar = pca_scale_input_covar.explained_variance_
            Eigen_df_scale_input_covar = pd.DataFrame(data=eigenvalues_scale_input_covar, columns=["Eigenvaues"])
            PC_df_scale_input_covar = pd.DataFrame(data=['PC' + str(i + 1) for i in range(len(features_input))],
                                                   columns=['Principal Component'])
            Var_df_scale_input_covar = pd.DataFrame(data=Var_scale_input_covar,
                                                    columns=['Cumulative Proportion of Explained Ratio'])
            Var_cumsum_scale_input_covar = Var_df_scale_input_covar.cumsum()
            Var_dfff_scale_input_covar = pd.concat([Var_cumsum_scale_input_covar * 100], axis=1)
            Eigen_Analysis_scale_input_covar = pd.concat([PC_df_scale_input_covar.T, Eigen_df_scale_input_covar.T,
                                                          Var_df_scale_input_covar.T, Var_dfff_scale_input_covar.T],
                                                         axis=0)
            Eigen_Analysis_scale_input_covar = Eigen_Analysis_scale_input_covar.rename(
                columns=Eigen_Analysis_scale_input_covar.iloc[0])
            Eigen_Analysis_scale_input_covar = Eigen_Analysis_scale_input_covar.drop(
                Eigen_Analysis_scale_input_covar.index[0])
            Eigen_Analysis_scale_input_covar.insert(loc=0, column="Principal Components",
                                                    value=["Eigenvalues", "Proportion of Explained Variance",
                                                           "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_scale_input_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            dff_target = dff[input]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
            # INPUT DATA WITH REMOVING OUTLIERS
            pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
                x_scale_input_outlier_covar)
            principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                                 , columns=['PC' + str(i + 1) for i in
                                                                            range(len(features_input_outlier))])
            finalDf_scale_input_outlier_covar = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
            Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
            eigenvalues_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_
            Eigen_df_scale_input_outlier_covar = pd.DataFrame(data=eigenvalues_scale_input_outlier_covar,
                                                              columns=["Eigenvaues"])
            PC_df_scale_input_outlier_covar = pd.DataFrame(
                data=['PC' + str(i + 1) for i in range(len(features_input_outlier))],
                columns=['Principal Component'])
            Var_df_scale_input_outlier_covar = pd.DataFrame(data=Var_scale_input_outlier_covar,
                                                            columns=['Cumulative Proportion of Explained '
                                                                     'Ratio'])
            Var_cumsum_scale_input_outlier_covar = Var_df_scale_input_outlier_covar.cumsum()
            Var_dfff_scale_input_outlier_covar = pd.concat([Var_cumsum_scale_input_outlier_covar * 100], axis=1)
            Eigen_Analysis_scale_input_outlier_covar = pd.concat(
                [PC_df_scale_input_outlier_covar.T, Eigen_df_scale_input_outlier_covar.T,
                 Var_df_scale_input_outlier_covar.T,
                 Var_dfff_scale_input_outlier_covar.T], axis=0)
            Eigen_Analysis_scale_input_outlier_covar = Eigen_Analysis_scale_input_outlier_covar.rename(
                columns=Eigen_Analysis_scale_input_outlier_covar.iloc[0])
            Eigen_Analysis_scale_input_outlier_covar = Eigen_Analysis_scale_input_outlier_covar.drop(
                Eigen_Analysis_scale_input_outlier_covar.index[0])
            Eigen_Analysis_scale_input_outlier_covar.insert(loc=0, column="Principal Components",
                                                            value=["Eigenvalues", "Proportion of Explained Variance",
                                                                   "Cumulative Proportion of Explained Variance (%)"])
            data_frame_EigenA = Eigen_Analysis_scale_input_outlier_covar
    data = data_frame_EigenA.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame_EigenA.columns]
    csv_string = data_frame_EigenA.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-loadings', 'download'),
              [Input('eigenA-outlier', 'value'),
               Input("matrix-type-data-table", 'value')])
def update_filename(outlier, matrix_type):
    if outlier == 'Yes' and matrix_type == "Correlation":
        download = 'Loadings_correlation_matrix_removed_outliers_data.csv'
    elif outlier == 'Yes' and matrix_type == "Covariance":
        download = 'Loadings_covariance_matrix_removed_outliers_data.csv'
    elif outlier == 'No' and matrix_type == "Correlation":
        download = 'Loadings_correlation_matrix_data.csv'
    elif outlier == 'No' and matrix_type == "Covariance":
        download = 'Loadings_covariance_matrix_data.csv'
    return download


@app.callback([Output('data-table-loadings', 'data'),
               Output('data-table-loadings', 'columns'),
               Output('download-link-loadings', 'href')],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input("matrix-type-data-table", 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, matrix_type, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        if outlier == 'No' and matrix_type == "Correlation":
            features1 = dff.columns
            features = list(features1)
            # ORIGINAL DATA WITH OUTLIERS
            x_scale = dff.loc[:, features].values
            y_scale = dff.loc[:, ].values
            x_scale = StandardScaler().fit_transform(x_scale)
            pca_scale = PCA(n_components=len(features))
            principalComponents_scale = pca_scale.fit_transform(x_scale)
            principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                             , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            # calculating loading vector plot
            loading_scale = pca_scale.components_.T * np.sqrt(pca_scale.explained_variance_)
            loading_scale_df = pd.DataFrame(data=loading_scale,
                                            columns=["PC" + str(i + 1) for i in range(len(features))])
            line_group_scale_df = pd.DataFrame(data=features, columns=['Features'])
            loading_scale_dataf = pd.concat([line_group_scale_df, loading_scale_df], axis=1)
            data_frame = loading_scale_dataf
        elif outlier == 'Yes' and matrix_type == "Correlation":
            # OUTLIER DATA
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]
            # ORIGINAL DATA WITH REMOVING OUTLIERS
            x_outlier_scale = outlier_dff.loc[:, features_outlier].values
            y_outlier_scale = outlier_dff.loc[:, ].values
            x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)
            # uses covariance matrix
            pca_outlier_scale = PCA(n_components=len(features_outlier))
            principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
            principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                     ,
                                                     columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
            # combining principle components and target
            loading_outlier_scale = pca_outlier_scale.components_.T * np.sqrt(pca_outlier_scale.explained_variance_)
            loading_outlier_scale_df = pd.DataFrame(data=loading_outlier_scale,
                                                    columns=["PC" + str(i + 1) for i in range(len(features_outlier))])
            line_group_outlier_scale_df = pd.DataFrame(data=features_outlier, columns=['Features'])
            loading_outlier_scale_dataf = pd.concat([line_group_outlier_scale_df, loading_outlier_scale_df], axis=1)
            data_frame = loading_outlier_scale_dataf
        elif outlier == "No" and matrix_type == "Covariance":
            features1 = dff.columns
            features = list(features1)
            x_scale_covar = dff.loc[:, features].values
            y_scale_covar = dff.loc[:, ].values
            pca_scale_covar = PCA(n_components=len(features))
            principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
            principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                                   , columns=['PC' + str(i + 1) for i in range(len(features))])
            loading_scale_covar = pca_scale_covar.components_.T * np.sqrt(pca_scale_covar.explained_variance_)
            loading_scale_df_covar = pd.DataFrame(data=loading_scale_covar,
                                                  columns=["PC" + str(i + 1) for i in range(len(features))])
            line_group_scale_df_covar = pd.DataFrame(data=features, columns=['Features'])
            loading_scale_dataf_covar = pd.concat([line_group_scale_df_covar, loading_scale_df_covar], axis=1)
            data_frame = loading_scale_dataf_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]
            # ORIGINAL DATA WITH REMOVING OUTLIERS
            x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
            y_outlier_scale_covar = outlier_dff.loc[:, ].values
            # uses covariance matrix
            pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
            principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
            principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar,
                                                           columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_outlier))])
            # combining principle components and target
            loading_outlier_scale_covar = pca_outlier_scale_covar.components_.T * np.sqrt(
                pca_outlier_scale_covar.explained_variance_)
            loading_outlier_scale_df_covar = pd.DataFrame(data=loading_outlier_scale_covar,
                                                          columns=["PC" + str(i + 1) for i in
                                                                   range(len(features_outlier))])
            line_group_outlier_scale_df_covar = pd.DataFrame(data=features_outlier, columns=['Features'])
            loading_outlier_scale_dataf_covar = pd.concat(
                [line_group_outlier_scale_df_covar, loading_outlier_scale_df_covar], axis=1)
            data_frame = loading_outlier_scale_dataf_covar
    if all_custom == 'Custom':
        if outlier == 'No' and matrix_type == "Correlation":
            # Dropping Data variables
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # INPUT DATA WITH OUTLIERS
            x_scale_input = dff_input.loc[:, features_input].values
            y_scale_input = dff_input.loc[:, ].values
            x_scale_input = StandardScaler().fit_transform(x_scale_input)

            pca_scale_input = PCA(n_components=len(features_input))
            principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
            principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                                   , columns=['PC' + str(i + 1) for i in range(len(features_input))])
            finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
            dfff_scale_input = finalDf_scale_input.fillna(0)
            Var_scale_input = pca_scale_input.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input = pca_scale_input.components_.T * np.sqrt(pca_scale_input.explained_variance_)
            loading_scale_input_df = pd.DataFrame(data=loading_scale_input,
                                                  columns=["PC" + str(i + 1) for i in range(len(features_input))])
            line_group_scale_input_df = pd.DataFrame(data=features_input, columns=['Features'])
            loading_scale_input_dataf = pd.concat([line_group_scale_input_df, loading_scale_input_df], axis=1)
            data_frame = loading_scale_input_dataf
        elif outlier == 'Yes' and matrix_type == "Correlation":
            dff_input = dff.drop(columns=dff[input])
            dff_target = dff[input]
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            # INPUT DATA WITH REMOVING OUTLIERS
            x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier = dff_input_outlier.loc[:, ].values
            x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)
            pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
            principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                           , columns=['PC' + str(i + 1) for i in
                                                                      range(len(features_input_outlier))])
            finalDf_scale_input_outlier = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
            Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_outlier = pca_scale_input_outlier.components_.T * np.sqrt(
                pca_scale_input_outlier.explained_variance_)
            loading_scale_input_outlier_df = pd.DataFrame(data=loading_scale_input_outlier,
                                                          columns=["PC" + str(i + 1)
                                                                   for i in range(len(features_input_outlier))])
            line_group_scale_input_outlier_df = pd.DataFrame(data=features_input_outlier, columns=['Features'])
            loading_scale_input_outlier_dataf = pd.concat([line_group_scale_input_outlier_df,
                                                           loading_scale_input_outlier_df], axis=1)
            data_frame = loading_scale_input_outlier_dataf
        elif outlier == "No" and matrix_type == "Covariance":
            # Dropping Data variables
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # INPUT DATA WITH OUTLIERS
            x_scale_input_covar = dff_input.loc[:, features_input].values
            y_scale_input_covar = dff_input.loc[:, ].values
            pca_scale_input_covar = PCA(n_components=len(features_input))
            principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
            principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                         , columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_input))])
            finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target],
                                                  axis=1)
            dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
            Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_covar = pca_scale_input_covar.components_.T * np.sqrt(
                pca_scale_input_covar.explained_variance_)
            loading_scale_input_df_covar = pd.DataFrame(data=loading_scale_input_covar,
                                                        columns=["PC" + str(i + 1) for i in range(len(features_input))])
            line_group_scale_input_df_covar = pd.DataFrame(data=features_input, columns=['Features'])
            loading_scale_input_dataf_covar = pd.concat([line_group_scale_input_df_covar, loading_scale_input_df_covar],
                                                        axis=1)
            data_frame = loading_scale_input_dataf_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            dff_target = dff[input]
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            # INPUT DATA WITH REMOVING OUTLIERS
            x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
            pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
                x_scale_input_outlier_covar)
            principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                                 , columns=['PC' + str(i + 1) for i in
                                                                            range(len(features_input_outlier))])
            finalDf_scale_input_outlier_covar = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
            Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_outlier_covar = pca_scale_input_outlier_covar.components_.T * np.sqrt(
                pca_scale_input_outlier_covar.explained_variance_)
            loading_scale_input_outlier_df_covar = pd.DataFrame(data=loading_scale_input_outlier_covar,
                                                                columns=["PC" + str(i + 1)
                                                                         for i in range(len(features_input_outlier))])
            line_group_scale_input_outlier_df_covar = pd.DataFrame(data=features_input_outlier, columns=['Features'])
            loading_scale_input_outlier_dataf_covar = pd.concat([line_group_scale_input_outlier_df_covar,
                                                                 loading_scale_input_outlier_df_covar], axis=1)
            data_frame = loading_scale_input_outlier_dataf_covar

    data = data_frame.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame.columns]
    csv_string = data_frame.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-cos2', 'download'),
              [Input('eigenA-outlier', 'value'),
               Input("matrix-type-data-table", 'value')])
def update_filename(outlier, matrix_type):
    if outlier == 'Yes' and matrix_type == "Correlation":
        download = 'Cos2_correlation_matrix_removed_outliers_data.csv'
    elif outlier == 'Yes' and matrix_type == "Covariance":
        download = 'Cos2_covariance_matrix_removed_outliers_data.csv'
    elif outlier == 'No' and matrix_type == "Correlation":
        download = 'Cos2_correlation_matrix_data.csv'
    elif outlier == "No" and matrix_type == "Covariance":
        download = 'Cos2_covariance_matrix_data.csv'
    return download


@app.callback([Output('data-table-cos2', 'data'),
               Output('data-table-cos2', 'columns'),
               Output('download-link-cos2', 'href'), ],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input("matrix-type-data-table", 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, matrix_type, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == "All":
        if outlier == 'No' and matrix_type == "Correlation":
            features1 = dff.columns
            features = list(features1)
            x_scale = dff.loc[:, features].values
            y_scale = dff.loc[:, ].values
            x_scale = StandardScaler().fit_transform(x_scale)
            pca_scale = PCA(n_components=len(features))
            principalComponents_scale = pca_scale.fit_transform(x_scale)
            principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                             , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            finalDf_scale = pd.concat([df[[df.columns[0]]], principalDf_scale], axis=1)
            Var_scale = pca_scale.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale = pca_scale.components_.T * np.sqrt(pca_scale.explained_variance_)
            loading_scale_df = pd.DataFrame(data=loading_scale,
                                            columns=["PC" + str(i + 1) for i in range(len(features))])
            for i in loading_scale_df.columns:
                loading_scale_df[i] = (loading_scale_df[i] ** 2)
            line_group_scale_df = pd.DataFrame(data=features, columns=['Features'])
            loading_scale_dataf = pd.concat([line_group_scale_df, loading_scale_df], axis=1)
            data_frame = loading_scale_dataf
        elif outlier == 'Yes' and matrix_type == "Correlation":
            # ORIGINAL DATA WITH REMOVING OUTLIERS
            # OUTLIER DATA
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]

            x_outlier_scale = outlier_dff.loc[:, features_outlier].values
            y_outlier_scale = outlier_dff.loc[:, ].values
            x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)

            pca_outlier_scale = PCA(n_components=len(features_outlier))
            principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
            principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                     ,
                                                     columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
            finalDf_outlier_scale = pd.concat([outlier_names, principalDf_outlier_scale], axis=1)
            Var_outlier_scale = pca_outlier_scale.explained_variance_ratio_

            loading_outlier_scale = pca_outlier_scale.components_.T * np.sqrt(pca_outlier_scale.explained_variance_)
            loading_outlier_scale_df = pd.DataFrame(data=loading_outlier_scale,
                                                    columns=["PC" + str(i + 1) for i in range(len(features_outlier))])

            for i in loading_outlier_scale_df.columns:
                loading_outlier_scale_df[i] = loading_outlier_scale_df[i] ** 2
            line_group_outlier_scale_df = pd.DataFrame(data=features_outlier, columns=['Features'])
            loading_outlier_scale_dataf = pd.concat([line_group_outlier_scale_df, loading_outlier_scale_df], axis=1)
            data_frame = loading_outlier_scale_dataf
        elif outlier == "No" and matrix_type == "Covariance":
            features1 = dff.columns
            features = list(features1)
            x_scale_covar = dff.loc[:, features].values
            y_scale_covar = dff.loc[:, ].values
            pca_scale_covar = PCA(n_components=len(features))
            principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
            principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                                   , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            finalDf_scale_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_covar], axis=1)
            Var_scale_covar = pca_scale_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_covar = pca_scale_covar.components_.T * np.sqrt(pca_scale_covar.explained_variance_)
            loading_scale_df_covar = pd.DataFrame(data=loading_scale_covar,
                                                  columns=["PC" + str(i + 1) for i in range(len(features))])
            for i in loading_scale_df_covar.columns:
                loading_scale_df_covar[i] = (loading_scale_df_covar[i] ** 2)
            line_group_scale_df_covar = pd.DataFrame(data=features, columns=['Features'])
            loading_scale_dataf_covar = pd.concat([line_group_scale_df_covar, loading_scale_df_covar], axis=1)
            data_frame = loading_scale_dataf_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            # OUTLIER DATA
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]

            x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
            y_outlier_scale_covar = outlier_dff.loc[:, ].values
            pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
            principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
            principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar,
                                                           columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_outlier))])
            finalDf_outlier_scale_covar = pd.concat([outlier_names, principalDf_outlier_scale_covar], axis=1)
            Var_outlier_scale_covar = pca_outlier_scale_covar.explained_variance_ratio_

            loading_outlier_scale_covar = pca_outlier_scale_covar.components_.T * np.sqrt(
                pca_outlier_scale_covar.explained_variance_)
            loading_outlier_scale_df_covar = pd.DataFrame(data=loading_outlier_scale_covar,
                                                          columns=["PC" + str(i + 1) for i in
                                                                   range(len(features_outlier))])

            for i in loading_outlier_scale_df_covar.columns:
                loading_outlier_scale_df_covar[i] = loading_outlier_scale_df_covar[i] ** 2
            line_group_outlier_scale_df_covar = pd.DataFrame(data=features_outlier, columns=['Features'])
            loading_outlier_scale_dataf_covar = pd.concat(
                [line_group_outlier_scale_df_covar, loading_outlier_scale_df_covar], axis=1)
            data_frame = loading_outlier_scale_dataf_covar
    if all_custom == 'Custom':
        if outlier == 'No' and matrix_type == "Correlation":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # INPUT DATA WITH OUTLIERS
            x_scale_input = dff_input.loc[:, features_input].values
            y_scale_input = dff_input.loc[:, ].values
            x_scale_input = StandardScaler().fit_transform(x_scale_input)

            pca_scale_input = PCA(n_components=len(features_input))
            principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
            principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                                   , columns=['PC' + str(i + 1) for i in range(len(features_input))])
            finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
            dfff_scale_input = finalDf_scale_input.fillna(0)
            Var_scale_input = pca_scale_input.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input = pca_scale_input.components_.T * np.sqrt(pca_scale_input.explained_variance_)
            loading_scale_input_df = pd.DataFrame(data=loading_scale_input,
                                                  columns=["PC" + str(i + 1) for i in range(len(features_input))])
            for i in loading_scale_input_df.columns:
                loading_scale_input_df[i] = loading_scale_input_df[i] ** 2
            line_group_scale_input_df = pd.DataFrame(data=features_input, columns=['Features'])
            loading_scale_input_dataf = pd.concat([line_group_scale_input_df, loading_scale_input_df], axis=1)
            data_frame = loading_scale_input_dataf
        elif outlier == "Yes" and matrix_type == "Correlation":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # OUTLIER DATA INPUT
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier = dff_input_outlier.loc[:, ].values
            x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)

            pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
            principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                           , columns=['PC' + str(i + 1) for i in
                                                                      range(len(features_input_outlier))])
            finalDf_scale_input_outlier = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
            Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_outlier = pca_scale_input_outlier.components_.T * np.sqrt(
                pca_scale_input_outlier.explained_variance_)
            loading_scale_input_outlier_df = pd.DataFrame(data=loading_scale_input_outlier,
                                                          columns=["PC" + str(i + 1) for i in
                                                                   range(len(features_input_outlier))])
            for i in loading_scale_input_outlier_df.columns:
                loading_scale_input_outlier_df[i] = (loading_scale_input_outlier_df[i] ** 2)
            line_group_scale_input_outlier_df = pd.DataFrame(data=features_input_outlier, columns=['Features'])
            loading_scale_input_outlier_dataf = pd.concat(
                [line_group_scale_input_outlier_df, loading_scale_input_outlier_df], axis=1)
            data_frame = loading_scale_input_outlier_dataf
        elif outlier == "No" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # INPUT DATA WITH OUTLIERS
            x_scale_input_covar = dff_input.loc[:, features_input].values
            y_scale_input_covar = dff_input.loc[:, ].values
            pca_scale_input_covar = PCA(n_components=len(features_input))
            principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
            principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                         , columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_input))])
            finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target],
                                                  axis=1)
            dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
            Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_covar = pca_scale_input_covar.components_.T * np.sqrt(
                pca_scale_input_covar.explained_variance_)
            loading_scale_input_df_covar = pd.DataFrame(data=loading_scale_input_covar,
                                                        columns=["PC" + str(i + 1) for i in range(len(features_input))])
            for i in loading_scale_input_df_covar.columns:
                loading_scale_input_df_covar[i] = loading_scale_input_df_covar[i] ** 2
            line_group_scale_input_df_covar = pd.DataFrame(data=features_input, columns=['Features'])
            loading_scale_input_dataf_covar = pd.concat([line_group_scale_input_df_covar, loading_scale_input_df_covar],
                                                        axis=1)
            data_frame = loading_scale_input_dataf_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # OUTLIER DATA INPUT
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
            pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
                x_scale_input_outlier_covar)
            principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                                 , columns=['PC' + str(i + 1) for i in
                                                                            range(len(features_input_outlier))])
            finalDf_scale_input_outlier_covar = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
            Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_outlier_covar = pca_scale_input_outlier_covar.components_.T * np.sqrt(
                pca_scale_input_outlier_covar.explained_variance_)
            loading_scale_input_outlier_df_covar = pd.DataFrame(data=loading_scale_input_outlier_covar,
                                                                columns=["PC" + str(i + 1) for i in
                                                                         range(len(features_input_outlier))])
            for i in loading_scale_input_outlier_df_covar.columns:
                loading_scale_input_outlier_df_covar[i] = (loading_scale_input_outlier_df_covar[i] ** 2)
            line_group_scale_input_outlier_df_covar = pd.DataFrame(data=features_input_outlier, columns=['Features'])
            loading_scale_input_outlier_dataf_covar = pd.concat(
                [line_group_scale_input_outlier_df_covar, loading_scale_input_outlier_df_covar], axis=1)
            data_frame = loading_scale_input_outlier_dataf_covar

    data = data_frame.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame.columns]
    csv_string = data_frame.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-contrib', 'download'),
              [Input('eigenA-outlier', 'value'),
               Input("matrix-type-data-table", 'value'), ])
def update_filename(outlier, matrix_type):
    if outlier == 'Yes' and matrix_type == "Correlation":
        download = 'Contributions_correlation_matrix_removed_outliers_data.csv'
    elif outlier == "Yes" and matrix_type == "Covariance":
        download = 'Contributions_covariance_matrix_removed_outliers_data.csv'
    elif outlier == 'No' and matrix_type == "Correlation":
        download = 'Contributions_correlation_matrix_data.csv'
    elif outlier == "No" and matrix_type == "Covariance":
        download = 'Contributions_covariance_matrix_data.csv'
    return download


@app.callback([Output('data-table-contrib', 'data'),
               Output('data-table-contrib', 'columns'),
               Output('download-link-contrib', 'href')],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input("matrix-type-data-table", 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, matrix_type, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == "All":
        if outlier == 'No' and matrix_type == "Correlation":
            features1 = dff.columns
            features = list(features1)
            x_scale = dff.loc[:, features].values
            y_scale = dff.loc[:, ].values
            x_scale = StandardScaler().fit_transform(x_scale)
            pca_scale = PCA(n_components=len(features))
            principalComponents_scale = pca_scale.fit_transform(x_scale)
            principalDf_scale = pd.DataFrame(data=principalComponents_scale
                                             , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            finalDf_scale = pd.concat([df[[df.columns[0]]], principalDf_scale], axis=1)
            Var_scale = pca_scale.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale = pca_scale.components_.T * np.sqrt(pca_scale.explained_variance_)
            loading_scale_df = pd.DataFrame(data=loading_scale,
                                            columns=["PC" + str(i + 1) for i in range(len(features))])
            for i in loading_scale_df.columns:
                loading_scale_df[i] = ((loading_scale_df[i] ** 2) * 100) / (loading_scale_df[i] ** 2).sum(axis=0)
            line_group_scale_df = pd.DataFrame(data=features, columns=['Features'])
            loading_scale_dataf = pd.concat([line_group_scale_df, loading_scale_df], axis=1)
            data_frame = loading_scale_dataf
        elif outlier == 'Yes' and matrix_type == "Correlation":
            # ORIGINAL DATA WITH REMOVING OUTLIERS
            # OUTLIER DATA
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]

            x_outlier_scale = outlier_dff.loc[:, features_outlier].values
            y_outlier_scale = outlier_dff.loc[:, ].values
            x_outlier_scale = StandardScaler().fit_transform(x_outlier_scale)

            pca_outlier_scale = PCA(n_components=len(features_outlier))
            principalComponents_outlier_scale = pca_outlier_scale.fit_transform(x_outlier_scale)
            principalDf_outlier_scale = pd.DataFrame(data=principalComponents_outlier_scale
                                                     ,
                                                     columns=['PC' + str(i + 1) for i in range(len(features_outlier))])
            finalDf_outlier_scale = pd.concat([outlier_names, principalDf_outlier_scale], axis=1)
            Var_outlier_scale = pca_outlier_scale.explained_variance_ratio_

            loading_outlier_scale = pca_outlier_scale.components_.T * np.sqrt(pca_outlier_scale.explained_variance_)
            loading_outlier_scale_df = pd.DataFrame(data=loading_outlier_scale,
                                                    columns=["PC" + str(i + 1) for i in range(len(features_outlier))])

            for i in loading_outlier_scale_df.columns:
                loading_outlier_scale_df[i] = ((loading_outlier_scale_df[i] ** 2) * 100) / (
                        loading_outlier_scale_df[i] ** 2).sum(axis=0)
            line_group_outlier_scale_df = pd.DataFrame(data=features_outlier, columns=['Features'])
            loading_outlier_scale_dataf = pd.concat([line_group_outlier_scale_df, loading_outlier_scale_df], axis=1)
            data_frame = loading_outlier_scale_dataf
        elif outlier == "No" and matrix_type == "Covariance":
            features1 = dff.columns
            features = list(features1)
            x_scale_covar = dff.loc[:, features].values
            y_scale_covar = dff.loc[:, ].values
            pca_scale_covar = PCA(n_components=len(features))
            principalComponents_scale_covar = pca_scale_covar.fit_transform(x_scale_covar)
            principalDf_scale_covar = pd.DataFrame(data=principalComponents_scale_covar
                                                   , columns=['PC' + str(i + 1) for i in range(len(features))])
            # combining principle components and target
            finalDf_scale_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_covar], axis=1)
            Var_scale_covar = pca_scale_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_covar = pca_scale_covar.components_.T * np.sqrt(pca_scale_covar.explained_variance_)
            loading_scale_df_covar = pd.DataFrame(data=loading_scale_covar,
                                                  columns=["PC" + str(i + 1) for i in range(len(features))])
            for i in loading_scale_df_covar.columns:
                loading_scale_df_covar[i] = ((loading_scale_df_covar[i] ** 2) * 100) / (
                        loading_scale_df_covar[i] ** 2).sum(axis=0)
            line_group_scale_df_covar = pd.DataFrame(data=features, columns=['Features'])
            loading_scale_dataf_covar = pd.concat([line_group_scale_df_covar, loading_scale_df_covar], axis=1)
            data_frame = loading_scale_dataf_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            z_scores = scipy.stats.zscore(dff)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            outlier_dff = dff[filtered_entries]
            features1_outlier = outlier_dff.columns
            features_outlier = list(features1_outlier)
            outlier_names1 = df[filtered_entries]
            outlier_names = outlier_names1.iloc[:, 0]

            x_outlier_scale_covar = outlier_dff.loc[:, features_outlier].values
            y_outlier_scale_covar = outlier_dff.loc[:, ].values
            pca_outlier_scale_covar = PCA(n_components=len(features_outlier))
            principalComponents_outlier_scale_covar = pca_outlier_scale_covar.fit_transform(x_outlier_scale_covar)
            principalDf_outlier_scale_covar = pd.DataFrame(data=principalComponents_outlier_scale_covar,
                                                           columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_outlier))])
            finalDf_outlier_scale_covar = pd.concat([outlier_names, principalDf_outlier_scale_covar], axis=1)
            Var_outlier_scale_covar = pca_outlier_scale_covar.explained_variance_ratio_

            loading_outlier_scale_covar = pca_outlier_scale_covar.components_.T * np.sqrt(
                pca_outlier_scale_covar.explained_variance_)
            loading_outlier_scale_df_covar = pd.DataFrame(data=loading_outlier_scale_covar,
                                                          columns=["PC" + str(i + 1) for i in
                                                                   range(len(features_outlier))])

            for i in loading_outlier_scale_df_covar.columns:
                loading_outlier_scale_df_covar[i] = ((loading_outlier_scale_df_covar[i] ** 2) * 100) / (
                        loading_outlier_scale_df_covar[i] ** 2).sum(axis=0)
            line_group_outlier_scale_df_covar = pd.DataFrame(data=features_outlier, columns=['Features'])
            loading_outlier_scale_dataf_covar = pd.concat(
                [line_group_outlier_scale_df_covar, loading_outlier_scale_df_covar], axis=1)
            data_frame = loading_outlier_scale_dataf_covar
    if all_custom == 'Custom':
        if outlier == 'No' and matrix_type == "Correlation":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # INPUT DATA WITH OUTLIERS
            x_scale_input = dff_input.loc[:, features_input].values
            y_scale_input = dff_input.loc[:, ].values
            x_scale_input = StandardScaler().fit_transform(x_scale_input)

            pca_scale_input = PCA(n_components=len(features_input))
            principalComponents_scale_input = pca_scale_input.fit_transform(x_scale_input)
            principalDf_scale_input = pd.DataFrame(data=principalComponents_scale_input
                                                   , columns=['PC' + str(i + 1) for i in range(len(features_input))])
            finalDf_scale_input = pd.concat([df[[df.columns[0]]], principalDf_scale_input, dff_target], axis=1)
            dfff_scale_input = finalDf_scale_input.fillna(0)
            Var_scale_input = pca_scale_input.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input = pca_scale_input.components_.T * np.sqrt(pca_scale_input.explained_variance_)
            loading_scale_input_df = pd.DataFrame(data=loading_scale_input,
                                                  columns=["PC" + str(i + 1) for i in range(len(features_input))])
            for i in loading_scale_input_df.columns:
                loading_scale_input_df[i] = ((loading_scale_input_df[i] ** 2) * 100) / (
                        loading_scale_input_df[i] ** 2).sum(axis=0)
            line_group_scale_input_df = pd.DataFrame(data=features_input, columns=['Features'])
            loading_scale_input_dataf = pd.concat([line_group_scale_input_df, loading_scale_input_df], axis=1)
            data_frame = loading_scale_input_dataf
        elif outlier == "Yes" and matrix_type == "Correlation":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # OUTLIER DATA INPUT
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            x_scale_input_outlier = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier = dff_input_outlier.loc[:, ].values
            x_scale_input_outlier = StandardScaler().fit_transform(x_scale_input_outlier)

            pca_scale_input_outlier = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier = pca_scale_input_outlier.fit_transform(x_scale_input_outlier)
            principalDf_scale_input_outlier = pd.DataFrame(data=principalComponents_scale_input_outlier
                                                           , columns=['PC' + str(i + 1) for i in
                                                                      range(len(features_input_outlier))])
            finalDf_scale_input_outlier = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier = finalDf_scale_input_outlier.fillna(0)
            Var_scale_input_outlier = pca_scale_input_outlier.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_outlier = pca_scale_input_outlier.components_.T * np.sqrt(
                pca_scale_input_outlier.explained_variance_)
            loading_scale_input_outlier_df = pd.DataFrame(data=loading_scale_input_outlier,
                                                          columns=["PC" + str(i + 1) for i in
                                                                   range(len(features_input_outlier))])
            for i in loading_scale_input_outlier_df.columns:
                loading_scale_input_outlier_df[i] = ((loading_scale_input_outlier_df[i] ** 2) * 100) / \
                                                    (loading_scale_input_outlier_df[i] ** 2).sum(axis=0)
            line_group_scale_input_outlier_df = pd.DataFrame(data=features_input_outlier, columns=['Features'])
            loading_scale_input_outlier_dataf = pd.concat(
                [line_group_scale_input_outlier_df, loading_scale_input_outlier_df], axis=1)
            data_frame = loading_scale_input_outlier_dataf
        elif outlier == "No" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # INPUT DATA WITH OUTLIERS
            x_scale_input_covar = dff_input.loc[:, features_input].values
            y_scale_input_covar = dff_input.loc[:, ].values
            pca_scale_input_covar = PCA(n_components=len(features_input))
            principalComponents_scale_input_covar = pca_scale_input_covar.fit_transform(x_scale_input_covar)
            principalDf_scale_input_covar = pd.DataFrame(data=principalComponents_scale_input_covar
                                                         , columns=['PC' + str(i + 1) for i in
                                                                    range(len(features_input))])
            finalDf_scale_input_covar = pd.concat([df[[df.columns[0]]], principalDf_scale_input_covar, dff_target],
                                                  axis=1)
            dfff_scale_input_covar = finalDf_scale_input_covar.fillna(0)
            Var_scale_input_covar = pca_scale_input_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_covar = pca_scale_input_covar.components_.T * np.sqrt(
                pca_scale_input_covar.explained_variance_)
            loading_scale_input_df_covar = pd.DataFrame(data=loading_scale_input_covar,
                                                        columns=["PC" + str(i + 1) for i in range(len(features_input))])
            for i in loading_scale_input_df_covar.columns:
                loading_scale_input_df_covar[i] = ((loading_scale_input_df_covar[i] ** 2) * 100) / (
                        loading_scale_input_df_covar[i] ** 2).sum(axis=0)
            line_group_scale_input_df_covar = pd.DataFrame(data=features_input, columns=['Features'])
            loading_scale_input_dataf_covar = pd.concat([line_group_scale_input_df_covar, loading_scale_input_df_covar],
                                                        axis=1)
            data_frame = loading_scale_input_dataf_covar
        elif outlier == "Yes" and matrix_type == "Covariance":
            dff_input = dff.drop(columns=dff[input])
            features1_input = dff_input.columns
            features_input = list(features1_input)
            dff_target = dff[input]
            # OUTLIER DATA INPUT
            z_scores_input = scipy.stats.zscore(dff_input)
            abs_z_scores_input = np.abs(z_scores_input)
            filtered_entries_input = (abs_z_scores_input < 3).all(axis=1)
            dff_input_outlier = dff_input[filtered_entries_input]
            features1_input_outlier = dff_input_outlier.columns
            features_input_outlier = list(features1_input_outlier)
            outlier_names_input1 = df[filtered_entries_input]
            outlier_names_input = outlier_names_input1.iloc[:, 0]
            # OUTLIER DATA TARGET
            z_scores_target = scipy.stats.zscore(dff_target)
            abs_z_scores_target = np.abs(z_scores_target)
            filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
            dff_target_outlier = dff_target[filtered_entries_target]
            x_scale_input_outlier_covar = dff_input_outlier.loc[:, features_input_outlier].values
            y_scale_input_outlier_covar = dff_input_outlier.loc[:, ].values
            pca_scale_input_outlier_covar = PCA(n_components=len(features_input_outlier))
            principalComponents_scale_input_outlier_covar = pca_scale_input_outlier_covar.fit_transform(
                x_scale_input_outlier_covar)
            principalDf_scale_input_outlier_covar = pd.DataFrame(data=principalComponents_scale_input_outlier_covar
                                                                 , columns=['PC' + str(i + 1) for i in
                                                                            range(len(features_input_outlier))])
            finalDf_scale_input_outlier_covar = pd.concat(
                [outlier_names_input, principalDf_scale_input_outlier_covar, dff_target_outlier],
                axis=1)
            dfff_scale_input_outlier_covar = finalDf_scale_input_outlier_covar.fillna(0)
            Var_scale_input_outlier_covar = pca_scale_input_outlier_covar.explained_variance_ratio_
            # calculating loading vector plot
            loading_scale_input_outlier_covar = pca_scale_input_outlier_covar.components_.T * np.sqrt(
                pca_scale_input_outlier_covar.explained_variance_)
            loading_scale_input_outlier_df_covar = pd.DataFrame(data=loading_scale_input_outlier_covar,
                                                                columns=["PC" + str(i + 1) for i in
                                                                         range(len(features_input_outlier))])
            for i in loading_scale_input_outlier_df_covar.columns:
                loading_scale_input_outlier_df_covar[i] = ((loading_scale_input_outlier_df_covar[i] ** 2) * 100) / \
                                                          (loading_scale_input_outlier_df_covar[i] ** 2).sum(axis=0)
            line_group_scale_input_outlier_df_covar = pd.DataFrame(data=features_input_outlier, columns=['Features'])
            loading_scale_input_outlier_dataf_covar = pd.concat(
                [line_group_scale_input_outlier_df_covar, loading_scale_input_outlier_df_covar], axis=1)
            data_frame = loading_scale_input_outlier_dataf_covar
    data = data_frame.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame.columns]
    csv_string = data_frame.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


# serve(server)
if __name__ == '__main__':
    # For Development only, otherwise use gunicorn or uwsgi to launch, e.g.
    # gunicorn -b 0.0.0.0:8050 index:app.server
    app.run_server()

# OUTPUT: YOU SHOULD USE AT LEAST X PRINCIPAL COMPONENTS (≥85% of explained variance)

