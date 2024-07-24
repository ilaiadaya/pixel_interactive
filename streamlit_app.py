import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Pixel_Importances',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :1234: Pixel Importances

This site interactively displays pixel importances for MNIST dataset, provided by modelling pixels in a CNN.
'''


pixel_importance_df = pd.read_csv('/workspaces/pixel_interactive/pixel_importance_1.csv')
pixel_importance_df["importance"] = 1-pixel_importance_df["acc"]
pixel_accuracy_dict = pixel_importance_df.set_index('pixel_id')['importance'].to_dict()
print(pixel_accuracy_dict.keys())

num_pixels = st.slider('Number of top pixels to show', min_value=1, max_value=100, value=12)

def show_important_pixels(pixel_importance_df, num_pixels=12, most_important=True):
    """
    Plots the average image with the top or bottom important pixels highlighted.

    Parameters:
    pixel_importance_df (DataFrame): DataFrame with pixel importance values.
    num_pixels (int): Number of top or bottom important pixels to highlight.
    most_important (bool): If True, highlights the most important pixels; if False, highlights the least important pixels.

    Returns:
    dict: Ordered dictionary of the top pixel values and their corresponding pixel numbers.
    """
    # Get the indices of the top or bottom important pixels
    if most_important:
        pixels = pixel_importance_df['importance'].nlargest(num_pixels).index
        color = 'Greens'
    else:
        pixels = pixel_importance_df['importance'].nsmallest(num_pixels).index
        color = 'Reds'

    # Create a mask for the pixels
    mask = np.zeros((28, 28))
    mask[np.unravel_index(pixels, mask.shape)] = pixel_importance_df['importance'].loc[pixels]

    # Create a dictionary of the top pixel values and their corresponding pixel numbers
    pixel_dict = pixel_importance_df.loc[pixels]['importance'].to_dict()

    # Order the dictionary by values
    pixel_dict = dict(sorted(pixel_dict.items(), key=lambda item: item[1], reverse=most_important))

    # Create a Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=mask,
        colorscale=color,
        hoverongaps = False))

    # Invert y-axis
    fig.update_yaxes(autorange="reversed")

    # Set aspect ratio
    fig.update_layout(
        autosize=False,
        width=600,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        )
    )
    st.plotly_chart(fig)

    return pixel_dict
pixel_dict = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True)

# Print the ordered dictionary
st.write(pixel_dict)
