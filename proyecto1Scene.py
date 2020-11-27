from gl import Render
from obj import Texture
from collections import namedtuple
from math import pi
import random

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

r = Render()
r.glInit(1000,1000)

r.lookAt(V3(1, 0, 1), V3(0, 0, 0), V3(0, 1, 0))
r.load('./sphere.obj', translate=(10,10,10), scale=(1.1,1,1.1), rotate=(0,0,0))
r.drawArrays('SHADER')


r.load('./sphere.obj', translate=(6,-3,6), scale=(0.1,0.1,0.1), rotate=(0,0,0))
r.drawArrays('SHADER')

r.load('./sphere.obj', translate=(5,0,6), scale=(0.1,0.1,0.1), rotate=(0,0,0))
r.drawArrays('SHADER')

r.load('./sphere.obj', translate=(4,-1,2), scale=(0.038,0.038,0.038), rotate=(0,0,0))
r.drawArrays('SHADER')

r.load('./sphere.obj', translate=(3,-1,2), scale=(0.03,0.03,0.03), rotate=(0,0,0))
r.drawArrays('SHADER')

r.load('./sphere.obj', translate=(4,-1,6), scale=(0.085,0.085,0.085), rotate=(0,0,0))
r.drawArrays('SHADER')

r.load('./sphere.obj', translate=(4,0,8), scale=(0.1,0.1,0.1), rotate=(0,0,0))
r.drawArrays('SHADER')

t = Texture('./sun2.bmp')
r.texture = t
r.light = V3(0,0,1)
r.lookAt(V3(1, 0, 1), V3(0, 0, 0), V3(0, 1, 0))
r.load('./Dark Knight.obj', translate=(9, 10, 20), scale=(0.045, 0.045, 0.045), rotate=(0,4*pi/5,pi))
r.drawArrays('GREYSHADER')

r.load('./cube.obj', translate=(0.5, 18, 20), scale=(3, 3.5, 1), rotate=(0,0,pi/2))
r.drawArrays('GREYSHADER')

t = Texture('./fur.bmp')
r.texture = t
r.load('./Alien Animal.obj', translate=(250, 0, 135), scale=(3, 3, 4), rotate=(pi, pi, 0))
r.drawArrays('TRIANGLE')

r.glFinish('proyectoScene.bmp')