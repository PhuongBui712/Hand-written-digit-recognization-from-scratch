import pygame as pg
pg.init()

font = pg.font.SysFont('arial', 20) 

class Button():
    def __init__(self, screen, x, y, width, height, buttonText='Button', onclickFunction=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onclickFunction = onclickFunction
        self.screen = screen

        self.fillColors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }

        self.buttonSurface = pg.Surface((self.width, self.height))
        self.buttonRect = pg.Rect(self.x, self.y, self.width, self.height)
        self.buttonSurf = font.render(buttonText, True, (20, 20, 20))

        # pressing attribute
        self.alreadyPressed = False
        self.inPressing = False

    def process(self):
        # In normal mode
        self.buttonSurface.fill(self.fillColors['normal'])
        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width/2 - self.buttonSurf.get_rect().width/2,
            self.buttonRect.height/2 - self.buttonSurf.get_rect().height/2
        ])

        self.screen.blit(self.buttonSurface, self.buttonRect)

        # clicked
        mousePos = pg.mouse.get_pos()
        
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])

            if pg.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])
                self.inPressing = True

            elif self.inPressing:
                self.onclickFunction()
                self.inPressing=False
