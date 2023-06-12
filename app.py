import pygame as pg
from pygame.locals import *
from button import Button
import sys
from neural_network import training, compute_nnet_output, add_ones

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

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
DEFAULT_LEFTUPPOINT = (1000, 1000)
DEFAULT_RIGHTBOTTOMPOINT = (-1, -1)

leftUpPoint = None
rightBottomPoint = None


objects = []

def clear_inputPart():
    global leftUpPoint, rightBottomPoint
    leftUpPoint = DEFAULT_LEFTUPPOINT
    rightBottomPoint = DEFAULT_RIGHTBOTTOMPOINT
    pg.draw.rect(screen, (0, 0, 0), pg.Rect(12, 12, 396, 396))


def predict():
    global leftUpPoint, rightBottomPoint
    img_arr = np.array(pg.PixelArray(screen))[leftUpPoint[0]:rightBottomPoint[0], leftUpPoint[1]:rightBottomPoint[1]].T.astype(np.float32)
    image = cv2.resize(img_arr, (28, 28))/255
    image = np.pad(image, (5, 5), 'constant', constant_values=0)
    image = cv2.resize(image, (28, 28))/255

    plt.imshow(image, cmap='gray')
    plt.show()

    Z = add_ones(image.reshape((1, image.shape[0]*image.shape[1])))

    global Ws
    predict_y = compute_nnet_output(Ws, Z)
    pg.draw.rect(screen, (0, 0, 0), pg.Rect(432, 40, 120, 200))
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
    global inputPart
    inputPart = pg.draw.rect(screen, (0, 0, 255), pg.Rect(10, 10, 400, 400),  2)

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
    leftUpPoint = DEFAULT_LEFTUPPOINT
    rightBottomPoint = DEFAULT_RIGHTBOTTOMPOINT
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
                if inputPart.collidepoint(cursor):
                    pg.draw.line(screen, (255, 255, 255), prev_cursor, cursor, width=20)

                    leftUpPoint = (min(leftUpPoint[0], cursor[0]-10), min(leftUpPoint[1], cursor[1]-10))
                    rightBottomPoint = (max(rightBottomPoint[0], cursor[0]+10), max(rightBottomPoint[1], cursor[1]+10))

                prev_cursor = cursor

        for obj in objects:
            obj.process()

        pg.display.update()