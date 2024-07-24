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
    page_icon=':1234:', # This is an emoji shortcode. Could be a URL too.
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

pixel_importance_df = pd.read_csv('pixel_importance_1.csv')
pixel_importance_df["importance"] = 1-pixel_importance_df["acc"]
pixel_importance_df = pixel_importance_df.drop(columns=['acc'])
# Move the 'importance' column to the position of the 'accuracy' column
cols = pixel_importance_df.columns.tolist()
cols.insert(3, cols.pop(cols.index('importance')))
pixel_importance_df = pixel_importance_df[cols]
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, all]

# Get the unique classes
classes = ['all'] + list(range(10))

# Create a slider for the number of top pixels to show
num_pixels = st.slider('Number of top pixels to show', min_value=1, max_value=100, value=12)

# Create tabs for each class
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([str(class_) for class_ in classes])

def show_important_pixels(pixel_importance_df, num_pixels=12, most_important=True,  val = "all"):
    """
    Plots the average image with the top or bottom important pixels highlighted.

    Parameters:
    pixel_importance_df (DataFrame): DataFrame with pixel importance values.
    num_pixels (int): Number of top or bottom important pixels to highlight.
    most_important (bool): If True, highlights the most important pixels; if False, highlights the least important pixels.

    Returns:
    DataFrame: DataFrame of the top pixel values and their corresponding pixel numbers and coordinates.
    """
    if val == "all":
    # Get the indices of the top or bottom important pixels
        if most_important:
            pixels = pixel_importance_df['importance'].nlargest(num_pixels).index
            color = 'Greens'
        else:
            pixels = pixel_importance_df['importance'].nsmallest(num_pixels).index
            color = 'Reds'
    else: 
        if most_important:
            name = "acc_class_" + str(val)
            pixels = pixel_importance_df[name].nlargest(num_pixels).index
            color = 'Greens'
        else:
            pixels = pixel_importance_df[name].nsmallest(num_pixels).index
            color = 'Reds'
    # Create a mask for the pixels
    mask = np.zeros((28, 28))
    mask[np.unravel_index(pixels, mask.shape)] = pixel_importance_df['importance'].loc[pixels]

    # Create a DataFrame of the top pixel values and their corresponding pixel numbers and coordinates
    pixel_df = pixel_importance_df.loc[pixels].reset_index()
    # Drop the 'accuracy' column


    # Order the DataFrame by importance
    pixel_df = pixel_df.sort_values('importance', ascending=not most_important)

    # Create a Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=mask,
        colorscale=color,
        hoverongaps = False,
        colorbar=dict(len=0.5, y=0.5)))  # Place colorbar at the top

    # Invert y-axis
    fig.update_yaxes(autorange="reversed")

    # Set aspect ratio and remove margins
    fig.update_layout(
        autosize=False,
        width=600,
        height=500,
        margin=dict(
            l=30,
            r=0,
            b=0,
            t=0,
            pad=0
        )
    )

    st.plotly_chart(fig)

    return pixel_df


with tab0:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val="all")

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)

with tab1:

        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=0)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)

with tab2:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=1)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)


with tab3:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=2)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)


with tab4:
    
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=3)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)


with tab5:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=4)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)

with tab6:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=5)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)
with tab7:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=6)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)
with tab8:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=7)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)
with tab9:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=8)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)
with tab10:
        
        # Show the important pixels for the current class
        pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True, val=9)

        # Order the DataFrame based on the selected method
        pixel_df = pixel_df.drop(columns=["index"])

        # Display the DataFrame
        st.dataframe(pixel_df)

# Get the pixel importance data
#pixel_df = show_important_pixels(pixel_importance_df, num_pixels=num_pixels, most_important=True)

# Create a dropdown menu for the ordering method
#order = st.selectbox('Order by:', ('importance', 'pixel_id', 'x', 'y'))

# Order the DataFrame based on the selected method
#pixel_df = pixel_df.drop(columns=["index"])

# Display the DataFrame
#st.dataframe(pixel_df)