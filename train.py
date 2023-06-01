import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import pickle
import cv2
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import img_to_array
# from architecture import model, model_2

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.applications import EfficientNetB3
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten

np.random.seed(42)


def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines', marker_color='#f29407'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines', marker_color='#0771f2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines', marker_color='#f29407'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines', marker_color='#0771f2'), row=2, col=1)

    fig.update_xaxes(title_text='Liczba epok', row=1, col=1)
    fig.update_xaxes(title_text='Liczba epok', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f"Metrics")

    po.plot(fig, filename=filename, auto_open=False)


# model config
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32
INPUT_SHAPE = (300, 300, 3)
