import pygame as pg
from pygame.locals import *
from button import Button
import sys
from neural_network import training, compute_nnet_output, add_ones

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from json import load
from glob import glob

#-----------------------------------------------------------------------------

def run_notebook(filename):
    with open(filename) as fp:
        nb = load(fp)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(line for line in cell['source'] if not line.startswith('%'))
            exec(source, globals(), locals())


def init_neuralNetwork():
    global Ws
    Ws = []
    if not exists('model/layer0.txt'):
        Ws = training()
    else:
        for path in glob('model/*'):
            weight = np.loadtxt(path)
            Ws.append(weight)

BACKGROUND_COLOR = (0, 0, 0)
SCREEN_SIZE = (600, 600)
INPUT_SIZE = (400, 400)
SCALED_SIZE = (28, 28)
DRAWING_COLOR = (255, 255, 255)

objects = []

def clear_inputPart():
    print('clear clicked')
    pg.draw.rect(screen, (0, 0, 0), pg.Rect(12, 12, 396, 396))


def predict():
    image = pg.Surface.copy(screen.subsurface(pg.Rect(12, 12, 396, 396)))
    image = pg.transform.scale(image, (28, 28))
    rgb_arr = np.rot90(np.flip(pg.surfarray.array3d(image), axis=1), 1)

    X = np.dot(rgb_arr[...,:3], [0.299, 0.587, 0.144])
    Z = add_ones(X.reshape((1, X.shape[0]*X.shape[1])))

    plt.imshow(X)
    plt.show()

    global Ws
    predict_y = compute_nnet_output(Ws, Z)
    pg.draw.rect(screen, (0, 0, 0), pg.Rect(432, 12, 146, 296))
    font = pg.font.SysFont('arial', 50)
    text = font.render(str(predict_y[0]), True, (255, 255, 255))
    text_rect = text.get_rect(center = predictionPart.center)
    screen.blit(text, text_rect)

def initialize_screen():
    # screen
    global screen
    screen = pg.display.set_mode(SCREEN_SIZE)
    screen.fill(BACKGROUND_COLOR)

    # input part
    pg.draw.rect(screen, (0, 0, 255), pg.Rect(10, 10, 400, 400),  2)

    # button
    predictButton = Button(screen, 110, 420, 60, 30, "Predict", predict)
    clearButton = Button(screen, 310, 420, 60, 30, "Clear", clear_inputPart)

    objects.append(predictButton)
    objects.append(clearButton)

    # predict part
    global predictionPart
    predictionPart = pg.draw.rect(screen, (0, 0, 255), pg.Rect(430, 10, 150, 300), 2)
    font = pg.font.SysFont('arial', 20)
    text = font.render("Prediction", True, (255, 255, 255))
    text_rect = text.get_rect(center = predictionPart.center)
    screen.blit(text, (text_rect[0], 20))

if __name__ == "__main__":
    pg.init()
    
    # Initialize window
    init_neuralNetwork()
    initialize_screen()

    # write digit
    drawing = False
    prev_cursor = None
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                prev_cursor = pg.mouse.get_pos()
                drawing=True

            if event.type == MOUSEBUTTONUP and event.button == 1:
                drawing = False


            if drawing and event.type == MOUSEMOTION:
                cursor = pg.mouse.get_pos()
                pg.draw.line(screen, (255, 255, 255), prev_cursor, cursor, width=30)
                
                prev_cursor = cursor

        for obj in objects:
            obj.process()

        pg.display.update()