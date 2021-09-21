#!/usr/bin/env python3
""" 
Author:
     Gutama Ibrahim Mohammad 
Created:  
       16/09/2021

"""
import os.path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import yaml
from joblib import load

from config import PLOTS_PATH


def shap_force_plot_local_tree(
        model,
        row,
        feature_names,
        out_names=None,
        link='identity',
        plot_cmap="PkYg",
        matplotlib=True,
        show=True,
        figsize=(10, 3),
        ordering_keys=None,
        ordering_keys_time_format=None,
        text_rotation=90,
        contribution_threshold=0.05):
    """

    Args:
        model: tree based machine learning model eg. RandomForest,DecisionTree, Xgboost
        row: (ndarray or pd.Dataframe),
              Matrix of feature values (# features) or (# samples x # features).
                 This provides the values of all the features,
                 and should be the same shape as the shap_values argument.

        feature_names: list
                         Containing features used to train the model
                         List of feature names (# features)
        out_names: str
                     The name of the output of the model
                         (plural to support multi-output plotting in the future).
        link: “identity” or “logit”
                 The transformation used when drawing the tick mark labels.
                 Using logit will change log-odds numbers into probabilities.
        plot_cmap:

        matplotlib:bool
                 Whether to use the default Javascript output, or the (less developed) matplotlib output.
                 Using matplotlib can be helpful in scenarios where rendering Javascript/HTML is inconvenient.
        show: bool
        figsize: tuple
        ordering_keys:
        ordering_keys_time_format:
        text_rotation:
        contribution_threshold:float
                 Controls the feature names/values that are displayed on force plot.
                 Only features that the magnitude of their shap value is
                  larger than min_perc * (sum of all abs shap values) will be displayed.

    Returns:None

    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    fig_force=shap.force_plot(base_value=explainer.expected_value[1],
                    shap_values=shap_values[1],
                    features=row,
                    feature_names=feature_names,
                    out_names=out_names,
                    link=link,
                    plot_cmap=plot_cmap,
                    matplotlib=matplotlib,
                    show=show,
                    figsize=figsize,
                    ordering_keys=ordering_keys,
                    ordering_keys_time_format=ordering_keys_time_format,
                    text_rotation=text_rotation,
                    contribution_threshold=contribution_threshold)
    path = str(PLOTS_PATH / 'shap_tree_force_plot.html')
    shap.save_html(path, fig_force)
    plt.savefig(PLOTS_PATH / "shap_tree_force_plot.png")



def shap_waterfall_plot_local(model,row,feature_names,max_display=15, show=True):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    fig=plt.figure(figsize=(20,10))
    fig_water=shap.plots._waterfall.waterfall_legacy(expected_value=explainer.expected_value[1], shap_values=shap_values[1],
                                           features=row,
                                           feature_names=feature_names, max_display=max_display, show=show)


    #fig.savefig(PLOTS_PATH / "shap_tree_waterfall_plot_local.png")

