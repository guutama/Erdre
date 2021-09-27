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


def vitualize_individual_prediction_tree_model_shap(
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
        contribution_threshold=0.05,
        max_display=15):
    """
    -------------------------------------------------------------
    Force plot gives us the explanation of a single model prediction.
    Red arrows in force-plot represent feature effects (SHAP values) that drive the prediction value higher while blue arrows are those effects that drive the prediction value lower.
     Each arrow’s size represents the magnitude of the corresponding feature’s effect.
      The “base value”  marks the model’s average prediction over the training set.
    The feature values for the largest effects are printed at the bottom of the plot.


----------------------------------------------------------------------------
    Waterfall is another local analysis plot of a single instance prediction
    The waterfall plot’s straight vertical line marks the model’s base value.
    The colored line is the prediction.
    Feature values are printed next to the prediction line for reference.
    Starting at the bottom of the plot, the prediction line shows how the SHAP values (i.e., the feature effects) accumulate from the base value to arrive at the model’s final score at the top of the plot.
     This is similar to a statistical linear model where the sum of effects, plus an intercept, equals the prediction.


    ----------------------------------------------------------------------------

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
    #fig = plt.figure(figsize=(20, 10))
    fig_force=shap.force_plot(base_value=explainer.expected_value[1],
                    shap_values=shap_values[1],
                    features=row,
                    feature_names=feature_names,
                    matplotlib=matplotlib,
                    out_names=out_names,
                    show=show,
                    link=link,
                    plot_cmap=plot_cmap,
                    figsize=figsize,
                    ordering_keys=ordering_keys,
                    ordering_keys_time_format=ordering_keys_time_format,
                    text_rotation=text_rotation,
                    contribution_threshold=contribution_threshold)




    shap.plots._waterfall.waterfall_legacy(expected_value=explainer.expected_value[1],
                                           shap_values=shap_values[1],
                                           features=row,
                                           feature_names=feature_names,
                                           max_display=max_display,
                                           show=show)

    shap.decision_plot(base_value=explainer.expected_value[1],
                       shap_values=shap_values[1],
                       link='logit',
                       features=row,
                       feature_names=(feature_names.tolist()),
                       show=show,
                       title="Decision Plot")

    path = str(PLOTS_PATH / 'shap_tree_force_plot.html')
    if show:
        plt.show()
    else:
        shap.save_html(path, fig_force)
        plt.savefig(PLOTS_PATH / "shap_tree_force_plot.png")
        #fig.savefig(PLOTS_PATH / "shap_tree_decision_plot_local.png")
        #fig.savefig(PLOTS_PATH / "shap_tree_decision_plot_local.png")





def visualize_feature_effects_global_tree_model_shap(model,
                                                     test_data,
                                                     feature_names,
                                                     class_names=["unworn", "worn"],
                                                     max_display=30,
                                                     show=True,
                                                     axis_color='#000000',
                                                     plot_type='violin',
                                                     layered_violin_max_num_bins=1000,
                                                     title="SHAP summary plot"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)
    shap.summary_plot(shap_values[1],
                      test_data,
                      feature_names=feature_names,
                      class_names=class_names,
                      max_display=max_display,
                      show=show,
                      axis_color=axis_color,
                      plot_type=plot_type,
                      layered_violin_max_num_bins=layered_violin_max_num_bins,
                      title=title)
    shap.summary_plot(shap_values,
                      test_data,
                      plot_type="bar",
                      feature_names=feature_names,
                      class_names=class_names,
                      title="Bar plot",
                      show=show)

    shap.summary_plot(shap_values[1],
                      test_data,
                      plot_type="bar",
                      feature_names=feature_names,
                      class_names=class_names[1],
                      title="Bar plot worn class",
                      show=show)

    if show:
        plt.show()



