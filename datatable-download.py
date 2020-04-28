import io
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import flask
import os
import pandas as pd
import urllib.parse
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import math
import scipy.stats
import dash_table
from dash_table.Format import Format, Scheme
from colour import Color

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)
file_path = "/Users/mythilisutharson/documents/cam_work/explorer_data/AAML_Oxygen_Data.csv"
df = pd.read_csv(file_path)
dff = df.select_dtypes(exclude=['object'])

app.layout = html.Div([
    html.Div([
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
    ]),
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
    ], style={'padding': 10}),
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
                  'width': '49%'}),
    html.A(
        'Download Data',
        id='download-link',
        download="rawdata.csv",
        href="",
        target="_blank"
    )
])


@app.callback(Output('feature-input', 'options'),
              [Input('all-custom-choice', 'value')])
def activate_input(all_custom):
    if all_custom == 'All':
        options = []
    elif all_custom == 'Custom':
        options = [{'label': i, 'value': i} for i in dff.columns]
    return options


@app.callback(Output('download-link', 'download'),
              [Input('all-custom-choice', 'value'),
               Input('outlier-value-biplot', 'value')])
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
               Input('outlier-value-biplot', 'value')])
def update_link(all_custom, input, outlier):
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


if __name__ == '__main__':
    app.run_server(debug=True)
