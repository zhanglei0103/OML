import os
import sys

import pandas as pd
import plotly_express as px
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# @st.cache(allow_output_mutation=True)
def feature_select_from_model_table(type_ml, targetvalue_column, select_df, features):
    if (type_ml == 'Supervised Classification') & (not select_df.empty):
        # st.write(ml_x)
        scatter_xaxis_model = LogisticRegression().fit(select_df[features], select_df[targetvalue_column])
        scatter_yaxis_model = RandomForestClassifier(random_state=42).fit(select_df[features], select_df[targetvalue_column])
        x_model = "Logistic Regression"
        y_model = "Random Forest"
    elif (type_ml == 'Supervised Regression') & (not select_df.empty):
        scatter_xaxis_model = LinearRegression().fit(select_df[features], select_df[targetvalue_column])
        scatter_yaxis_model = RandomForestRegressor(random_state=42).fit(select_df[features], select_df[targetvalue_column])
        x_model = "Linear Regression"
        y_model = "Random Forest"
    else:
        scatter_xaxis_model = None
        scatter_yaxis_model = None
        x_model = None
        y_model = None

    scatter_feature_im = pd.DataFrame()
    if (scatter_xaxis_model is not None) & (scatter_yaxis_model is not None):
        scatter_feature_im['Feature Name'] = features
        if len(scatter_xaxis_model.coef_.shape) == 2:
            # 2D of coef
            if scatter_xaxis_model.coef_.shape[0] > 1:
                sum_coef = np.sum(scatter_xaxis_model.coef_.__abs__(), axis=0)
                scatter_feature_im[x_model] = sum_coef
            else:
                scatter_feature_im[x_model] = scatter_xaxis_model.coef_[0]
        else:
            # 1D of coef
            scatter_feature_im[x_model] = scatter_xaxis_model.coef_
        scatter_feature_im[f"Rank of {x_model}"] = scatter_feature_im[x_model].abs().rank(pct=True)
        scatter_feature_im[y_model] = scatter_yaxis_model.feature_importances_
        scatter_feature_im[f"Rank of {y_model}"] = scatter_feature_im[y_model].abs().rank(pct=True)
        show_marker_text = [scatter_feature_im.loc[i, 'Feature Name'] if (scatter_feature_im.loc[i, f"Rank of {x_model}"] > 0.75) | (scatter_feature_im.loc[i, f"Rank of {y_model}"] > 0.75) else np.nan for i in scatter_feature_im.index]
        scatter_feature_importance_fig = px.scatter(scatter_feature_im, x=x_model, y=y_model, color='Feature Name',
                                                    text=show_marker_text,
                                                    title=f"Feature Importance Preview: {x_model} vs. {y_model}")
        scatter_feature_importance_fig.update_layout(
            title_x=0, title_font_size=20, title_font_color='#264653',
            margin=dict(l=0, r=0, b=10, t=80),
            showlegend=True,
            height=600,
            width=1200,
            plot_bgcolor='#F0F2F6',
        )
        scatter_feature_importance_fig.update_traces(marker_size=10)
    return scatter_feature_im, scatter_feature_importance_fig, x_model, y_model