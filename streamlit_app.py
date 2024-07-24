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
st.markdown("""
# :1234: Pixel Importances
This site interactively displays pixel importances for MNIST dataset, provided by modelling pixels in a CNN (for 10% of the dataset).
        
""", unsafe_allow_html=True)

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
show_background = st.checkbox('Show average image in the background')

# Create tabs for each class
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([str(class_) for class_ in classes])




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, accuracy_score
import struct

def load_mnist_data():
    def load_mnist_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return pd.DataFrame(images.reshape(num, -1))

    def load_mnist_labels(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return pd.Series(labels)

    train_images = load_mnist_images('train-images.idx3-ubyte')
    train_labels = load_mnist_labels('train-labels.idx1-ubyte')
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

    X = pd.concat([train_images, test_images], ignore_index=True)
    y = pd.concat([train_labels, test_labels], ignore_index=True)

    return {"data": X, "target": y.astype(np.uint8)}

# Usage:
mnist = load_mnist_data()
X, y = mnist["data"], mnist["target"]
X_nump = X.to_numpy().reshape(70000, 28, 28)


#### ----------------- Boilerplate Code functions ----------------- ####
def prepare_data(X, y, test_size=0.2, subset_fraction=0.2, num_classes=10, random_state=42):
    """
    This function splits the data into training and testing sets, and then creates a subset
    of the training data. It also converts the labels to categorical format.

    Args:
    - X (DataFrame or ndarray): The input features.
    - y (DataFrame or ndarray): The input labels.
    - test_size (float): The proportion of the dataset to include in the test split.
    - subset_fraction (float): The fraction of the training data to use as a subset.
    - num_classes (int): The number of classes for categorical conversion.
    - random_state (int): The seed used by the random number generator.

    Returns:
    - tuple: A tuple containing the training data, test data, training labels,
             test labels, training subset data, and training subset labels.
    """
    # Splitting the data into training and testing sets
    X_train, X_test, y_train_prev, y_test_prev = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert the labels to categorical format
    y_train = to_categorical(y_train_prev, num_classes)
    y_test = to_categorical(y_test_prev, num_classes)

    # Determine the split index for the training subset
    split_index = int(len(X_train) * subset_fraction)

    # Use a subset of the training data
    X_train_subset = X_train[:split_index].copy()  # Create a copy of the subset
    y_train_subset = y_train[:split_index]

    return X_train, X_test, y_train, y_test, X_train_subset, y_train_subset

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
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=mask,
        colorscale=color,
        hoverongaps = False,
        colorbar=dict(len=0.5, y=0.5))) 
    # Add the average image to the background if the toggle is turned on
    if show_background:
        if val == "all":
            average_image = X_nump.mean(axis=0)
        else: 
            average_image = X_nump[y == val].mean(axis=0)
        fig.add_trace(go.Heatmap(
            z=average_image,
            colorscale='gray',
            showscale=False,
            opacity=0.5
        ))


    # Invert y-axis
    fig.update_yaxes(autorange="reversed")

    # Set aspect ratio and remove margins

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
        ),
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            fixedrange=True,  # Disable panning and zooming
            range=[0, 28]  # Set range from 0 to 30
        ),
        yaxis=dict(
            fixedrange=True,  # Disable panning and zooming
            range=[0, 28]  # Set range from 0 to 30
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

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

