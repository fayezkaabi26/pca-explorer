import base64
import io
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
                        "https://codepen.io/sutharson/pen/dyYzEGZ.css"]

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
tabs_styles = {'height': '40px', 'font-family': 'Arial', 'fontSize': 14}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '3px solid #4a4a4a',
    'borderBottom': '1px solid #d6d6d6 ',
    'backgroundColor': '#f6f6f6',
    'color': '#4a4a4a',
    # 'fontColor': '#004a4a',
    'fontWeight': 'bold',
    'padding': '6px'
}


####################
# APP LAYOUT #
####################
fig = go.Figure()
fig1 = go.Figure()
app.layout = html.Div([
    html.Div([
        html.Img(
            src='https://raw.githubusercontent.com/aaml-analytics/mof-explorer/master/UOC.png',
            height='35', width='135', style={'display': 'inline-block', 'padding-left': '1%'}),
        html.Img(src='https://raw.githubusercontent.com/aaml-analytics/mof-explorer/master/A2ML-logo.png',
                 height='50', width='125', style={'float': 'right', 'display': 'inline-block', 'padding-right': '2%'}),
        html.H1("Principal Component Analysis Visualisation Tools",
                style={'display': 'inline-block', 'padding-left': '11%', 'text-align': 'center', 'fontSize': 36,
                       'color': 'white', 'font-family': 'Georgia'}),
        html.H1("...", style={'fontColor': '#3c3c3c', 'fontSize': 6})
    ], style={'backgroundColor': '#3d0027'}),
    html.Div([html.A('Refresh', href='/')], style={}),
    html.Div([
        html.H2("Upload Data", style={'fontSize': 24, 'font-family': 'Arial', 'color': '#3d0027'}, ),
        html.H3("Upload .txt, .csv or .xls files to starting exploring data...", style={'fontSize': 16,
                                                                                        'font-family': 'Arial'}),
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
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        "Upload Error!"),
                    dbc.ModalBody(
                        "Please upload a .txt, .csv or .xls file."),
                    dbc.ModalFooter(
                        dbc.Button("Close",
                                   id="close-upload",
                                   className="ml-auto")
                    ),
                ],
                id="modal-upload",
                is_open=False,
                centered=True,
                size="xl"
            )
        ]), ], style={'display': 'inline-block', 'padding-left': '1%', }),
    html.Div([dcc.Tabs([dcc.Tab(label='Scree Plot', style=tab_style, selected_style=tab_selected_style,
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
                                    html.Div([
                                        html.Label(["You should use attempt to use at least..."

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
                                    html.Div([
                                        html.Label(["Note: Data has been standardised (scaled)"])
                                    ], style={'padding-left': '1%'})
                                ]),
                        dcc.Tab(label='Feature correlation', style=tab_style,
                                selected_style=tab_selected_style,
                                children=[html.Div([html.Div([dcc.Graph(id='PC-feature-heatmap')
                                                              ], style={'width': '44%',
                                                                        'display': 'inline-block',
                                                                        'float': 'right'}),
                                                    html.Div([dcc.Graph(id='feature-heatmap')
                                                              ], style={'width': '54%',
                                                                        'display': 'inline-block',
                                                                        'float': 'left'}),
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
                                                                  'width': '49%', 'padding-left': '1%'}),
                                                    html.Div([html.Label(["Select color scale:",
                                                                          dcc.RadioItems(
                                                                              id='colorscale',
                                                                              options=[{'label': i, 'value': i}
                                                                                       for i in
                                                                                       ['Viridis', 'Plasma']],
                                                                              value='Plasma'
                                                                          )]),
                                                              ], style={'display': 'inline-block',
                                                                        'width': '49%', 'padding-left': '1%'}),
                                                    html.Div([
                                                        html.P("There are usually two ways multicollinearity, "
                                                               "which is when there are a number of variables "
                                                               "that are highly correlated, is dealt with "
                                                               "before analysis:"),
                                                        html.P("1) Use PCA to obtain a set of orthogonal ("
                                                               "not correlated) variables to analyse."),
                                                        html.P("2) Use correlation of determination (R²) to "
                                                               "determine which variables are highly "
                                                               "correlated and use only 1 in analysis. "
                                                               "Cut off for highly correlated variables "
                                                               "is ~0.7."),
                                                        html.P(
                                                            "In any case, it depends on the machine learning algorithm you may apply later. For correlation robust algorithms,"
                                                            " such as Random Forest, correlation of features will not be a concern. For non-correlation robust algorithms such as Linear Regression, "
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
                                        html.P("For custom variables..."),
                                        html.P(
                                            " Input variables you would not like as features in your PCA:"),
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
                                                                           'width': '49%', 'padding-left': '1%'}),
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
                                                                       'width': '49%', 'padding-left': '1%'}),
                                                             html.Div([
                                                                 html.P(
                                                                     "Note that PCA is an unsupervised technique. It only "
                                                                     "looks at the input features and does not take "
                                                                     "into account the output or the target"
                                                                     " (response) variable.")
                                                             ], style={'padding-left': '1%'}),
                                                             html.Div([
                                                                 html.P("For custom variables..."),
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
                                                                 html.P(),
                                                                 html.P(
                                                                     "A loading plot shows how "
                                                                     "strongly each characteristic (variable)"
                                                                     " influences a principal component. The angles between the vectors"
                                                                     " tell us how characteristics correlate with one another... "),
                                                                 html.P(
                                                                     "1) When two vectors are close, forming a small angle, the two "
                                                                     "variables they represent are positively correlated. "
                                                                     "2) If they meet each other at 90°, they are not likely to be correlated. "
                                                                     "3) When they diverge and form a large angle (close to 180°), they are negative correlated."),
                                                                 html.P(
                                                                     "The Score Plot involves the projection of the data onto the PCs in two dimensions."
                                                                     "The plot contains the original date but in the rotated (PC) coordinate system"),
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
                                        'Download PCA Data',
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
                                                                                      'display': 'inline-block', }),
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
              ])])


# READ FILE
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+'
                             )
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df


# SIZE MODAL CALLBACK UPLOAD FILE
@app.callback(
    [Output('modal-upload', 'is_open'),
     Output('output-data-upload', 'children')],
    [
        Input('data-table-upload', 'contents'),
        Input('close-upload', 'n_clicks')],
    [State('data-table-upload', 'filename')])
def update_output(contents, modal_close, filename):
    ctx = dash.callback_context
    user_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
    df = parse_contents(contents, filename)
    if not user_clicked or user_clicked == 'close':
        return dash.no_update, False

    if contents is None:
        return [], False

    if not filename.endswith(('.xls', '.csv', '.txt')):
        return [], True
    return df


@app.callback(Output('csv-data', 'data'),
              [Input('data-table-upload', 'contents')],
              [State('data-table-upload', 'filename')])
def parse_uploaded_file(contents, filename):
    if not filename:
        return dash.no_update
    df = parse_contents(contents, filename)
    return df.to_json(date_format='iso', orient='split')


@app.callback(Output('PC-Var-plot', 'figure'),
              [Input('outlier-value', 'value'),
               Input('csv-data', 'data')],
              )
def update_graph_stat(outlier, data):
    traces = []
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No':
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
    elif outlier == 'Yes':
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
    traces.append(go.Scatter(x=data['Principal Component'], y=data['Cumulative Proportion of Explained Variance'],
                             mode='lines', line=dict(color='Red')))
    return {'data': traces,

            'layout': go.Layout(title='<b>Cumulative Scree Plot Proportion of Explained Variance</b>',
                                titlefont=dict(family='Georgia', size=16),
                                xaxis={'title': 'Principal Component'}, yaxis={'title': 'Cumulative Explained Variance',
                                                                               'range': [0, 100]},
                                hovermode='closest', font=dict(family="Helvetica"), template="simple_white")
            }


@app.callback(
    Output('var-output-container-filter', 'children'),
    [Input('outlier-value', 'value'),
     Input('csv-data', 'data')],
)
def update_output(outlier, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No':
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
    elif outlier == 'Yes':
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


@app.callback(Output('PC-Eigen-plot', 'figure'),
              [Input('outlier-value', 'value'),
               Input('csv-data', 'data')]
              )
def update_graph_stat(outlier, data):
    traces = []
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if outlier == 'No':
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
    elif outlier == 'Yes':
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
    traces.append(go.Scatter(x=data['Principal Component'], y=data['Eigenvalues'], mode='lines'))
    return {'data': traces,

            'layout': go.Layout(title='<b>Scree Plot Eigenvalues</b>', xaxis={'title': 'Principal Component'},
                                titlefont=dict(family='Georgia', size=16),
                                yaxis={'title': 'Eigenvalues'}, hovermode='closest',
                                font=dict(family="Helvetica"), template="simple_white", )
            }


@app.callback(Output('PC-feature-heatmap', 'figure'),
              [
                  Input('PC-feature-outlier-value', 'value'),
                  Input('colorscale', 'value'),
                  Input('csv-data', 'data')]
              )
def update_graph_stat(outlier, colorscale, data):
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
    if outlier == 'No':
        data = loading_dff
    elif outlier == 'Yes':
        data = loading_dff_outlier_hm
    traces.append(go.Heatmap(
        z=data, x=features_outlier2, y=['PC' + str(i + 1) for i in range(loading_outlier_hm.shape[1])],
        colorscale="Viridis" if colorscale == 'Viridis' else "Plasma",
        # coord: represent the correlation between the various feature and the principal component itself
        colorbar={"title": "Loading"}))
    return {'data': traces,
            'layout': go.Layout(title='<b>PC and Feature Correlation Analysis</b>', xaxis={'title': 'Features'},
                                titlefont=dict(family='Georgia', size=16),
                                yaxis={'title': 'Principal Component'},
                                hovermode='closest', margin={'b': 110, 't': 50, 'l': 50},
                                font=dict(family="Helvetica", size=11)),
            }


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
        colorbar={"title": "R²"}))
    return {'data': traces,
            'layout': go.Layout(title='<b>Feature Correlation Analysis</b>', xaxis={},
                                titlefont=dict(family='Georgia', size=16),
                                yaxis={},
                                hovermode='closest', margin={'b': 110, 't': 50, 'l': 170, 'r': 50},
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
               Input('csv-data', 'data')],)
def populate_color_dropdown(input, target, outlier, graph_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    dff_target = dff[input]
    z_scores_target = scipy.stats.zscore(dff_target)
    abs_z_scores_target = np.abs(z_scores_target)
    filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
    dff_target_outlier = dff_target[filtered_entries_target]
    if target == 'Yes' and outlier == 'Yes' and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target_outlier.columns]
    elif target == 'Yes' and outlier == 'No' and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target.columns]
    elif target == 'No' or graph_type == 'Loadings':
        options = []
    return options


@app.callback(Output('size-scale-scores', 'options'),
              [Input('feature-input', 'value'),
               Input('radio-target-item-second', 'value'),
               Input('outlier-value-biplot', 'value'),
               Input('customvar-graph-update', 'value'),
               Input('csv-data', 'data')])
def populate_color_dropdown(input, target, outlier, graph_type, data):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    dff_target = dff[input]
    z_scores_target = scipy.stats.zscore(dff_target)
    abs_z_scores_target = np.abs(z_scores_target)
    filtered_entries_target = (abs_z_scores_target < 3).all(axis=1)
    dff_target_outlier = dff_target[filtered_entries_target]
    if target == 'Yes' and outlier == 'Yes' and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target_outlier.columns]
    elif target == 'Yes' and outlier == 'No' and graph_type == 'Biplot':
        options = [{'label': i, 'value': i} for i in dff_target.columns]
    elif target == 'No' or graph_type == 'Loadings':
        options = []
    return options


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
                  Input('csv-data', 'data')
              ]
              )
def update_graph_custom(outlier, input, graph_update, color, target, size, target2, all_custom, data):
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
        if outlier == 'No':
            dat = dfff_scale
        elif outlier == 'Yes':
            dat = dfff_outlier_scale
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
        if outlier == 'No':
            data = loading_scale_line_graph
            variance = Var_scale
        elif outlier == 'Yes':
            data = loading_outlier_scale_line_graph
            variance = Var_outlier_scale
        counter = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf_all = data[data['line_group'] == i]
            trace1_all = go.Scatter(x=dataf_all['PC1'], y=dataf_all['PC2'], line=dict(color="#4f4f4f"),
                                    name=i,
                                    # text=i,
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
        if outlier == 'No':
            dat = dfff_scale_input
            variance = Var_scale_input
        elif outlier == 'Yes':
            dat = dfff_scale_input_outlier
            variance = Var_scale_input_outlier
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
        if outlier == 'No':
            data = loading_scale_input_line_graph
        elif outlier == 'Yes':
            data = loading_scale_input_outlier_line_graph
        counter = 0
        lists = [[] for i in range(len(data['line_group'].unique()))]
        for i in data['line_group'].unique():
            dataf = data[data['line_group'] == i]
            trace1 = go.Scatter(x=dataf['PC1'], y=dataf['PC2'],
                                line=dict(color="#666666" if target == 'Yes' else '#4f4f4f'), name=i,
                                # text=i,
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
                  Input('csv-data', 'data')
              ])
def update_cos2_plot(outlier, input, all_custom, data):
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
        # scaling data
        if outlier == 'No':
            data = loading_scale_line_graph_sort
            variance = Var_scale
        elif outlier == 'Yes':
            data = loading_outlier_scale_line_graph_sort
            variance = Var_outlier_scale
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
        ####################################################################################################
        if outlier == 'No':
            data = loading_scale_input_line_graph_sort
            variance = Var_scale_input
        elif outlier == 'Yes':
            variance = Var_scale_input_outlier
            data = loading_scale_input_outlier_line_graph_sort
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
                  Input('csv-data', 'data')
              ])
def update_cos2_plot(outlier, input, all_custom, data):
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
        # scaling data
        if outlier == 'No':
            data = loading_scale_line_graph_sort
            variance = Var_scale
        elif outlier == 'Yes':
            data = loading_outlier_scale_line_graph_sort
            variance = Var_outlier_scale
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
                                    textposition='bottom right', textfont=dict(size=12)
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
        ####################################################################################################
        if outlier == 'No':
            data = loading_scale_input_line_graph_sort
            variance = Var_scale_input
        elif outlier == 'Yes':
            variance = Var_scale_input_outlier
            data = loading_scale_input_outlier_line_graph_sort
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
               Input('eigenA-outlier', 'value')])
def update_filename(all_custom, outlier):
    if all_custom == 'All' and outlier == 'Yes':
        download = 'all_variables_outliers_removed_data.csv'
    elif all_custom == 'All' and outlier == 'No':
        download = 'all_variables_data.csv'
    elif all_custom == 'Custom' and outlier == 'Yes':
        download = 'custom_variables_outliers_removed_data.csv'
    elif all_custom == 'Custom' and outlier == 'No':
        download = 'custom_variables.csv'
    return download


@app.callback(Output('download-link', 'href'),
              [Input('all-custom-choice', 'value'),
               Input('feature-input', 'value'),
               Input('eigenA-outlier', 'value'),
               Input('csv-data', 'data')])
def update_link(all_custom, input, outlier, data):
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
        if outlier == 'No':
            dat = dfff_scale
        elif outlier == 'Yes':
            dat = dfff_outlier_scale
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
        if outlier == 'No':
            dat = dfff_scale_input
        elif outlier == 'Yes':
            dat = dfff_scale_input_outlier
    csv_string = dat.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return csv_string


@app.callback(Output('download-link-correlation', 'download'),
              [Input('eigenA-outlier', 'value')])
def update_filename(outlier):
    if outlier == 'Yes':
        download = 'feature_correlation_data_removed_outliers.csv'
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
              [
                  Input('eigenA-outlier', 'value')])
def update_filename(outlier):
    if outlier == 'Yes':
        download = 'Eigen_Analysis_data_removed_outliers.csv'
    elif outlier == 'No':
        download = 'Eigen_Analysis_data.csv'
    return download


@app.callback([Output('data-table-eigenA', 'data'),
               Output('data-table-eigenA', 'columns'),
               Output('download-link-eigenA', 'href')],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        if outlier == 'No':
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

    elif all_custom == "Custom":
        if outlier == 'No':
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
        if outlier == "Yes":
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

    data = data_frame_EigenA.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame_EigenA.columns]
    csv_string = data_frame_EigenA.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-loadings', 'download'),
              [
                  Input('eigenA-outlier', 'value')])
def update_filename(outlier):
    if outlier == 'Yes':
        download = 'Loadings_data_removed_outliers.csv'
    elif outlier == 'No':
        download = 'Loadings_data.csv'
    return download


@app.callback([Output('data-table-loadings', 'data'),
               Output('data-table-loadings', 'columns'),
               Output('download-link-loadings', 'href')],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == 'All':
        if outlier == 'No':
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
        if outlier == 'Yes':
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
    if all_custom == 'Custom':
        if outlier == 'No':
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
        if outlier == 'Yes':
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

    data = data_frame.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame.columns]
    csv_string = data_frame.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-cos2', 'download'),
              [Input('eigenA-outlier', 'value')])
def update_filename(outlier):
    if outlier == 'Yes':
        download = 'Cos2_data_removed_outliers.csv'
    elif outlier == 'No':
        download = 'Cos2_data.csv'
    return download


@app.callback([Output('data-table-cos2', 'data'),
               Output('data-table-cos2', 'columns'),
               Output('download-link-cos2', 'href'), ],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == "All":
        if outlier == 'No':
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
        if outlier == 'Yes':
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
    if all_custom == 'Custom':
        if outlier == 'No':
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
        if outlier == "Yes":
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

    data = data_frame.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in data_frame.columns]
    csv_string = data_frame.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-contrib', 'download'),
              [Input('eigenA-outlier', 'value')])
def update_filename(outlier):
    if outlier == 'Yes':
        download = 'Contributions_data_removed_outliers.csv'
    elif outlier == 'No':
        download = 'Contributions_data.csv'
    return download


@app.callback([Output('data-table-contrib', 'data'),
               Output('data-table-contrib', 'columns'),
               Output('download-link-contrib', 'href')],
              [Input('all-custom-choice', 'value'),
               Input("eigenA-outlier", 'value'),
               Input('feature-input', 'value'),
               Input('csv-data', 'data')], )
def update_output(all_custom, outlier, input, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    dff = df.select_dtypes(exclude=['object'])
    if all_custom == "All":
        if outlier == 'No':
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
        if outlier == 'Yes':
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
    if all_custom == 'Custom':
        if outlier == 'No':
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
        if outlier == "Yes":
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
    app.run_server(debug=False)

# OUTPUT: YOU SHOULD USE AT LEAST X PRINCIPAL COMPONENTS (≥85% of explained variance)
