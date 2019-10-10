#!/usr/bin/python

import gpl
from gpl import *

renderer = gpl.cvar.Canvas_s_renderer
renderer.init(800, 720, "TEST")

primitives  = [Point(vec3((i+1)*100,       (i+1)*100, 0), vec4(  i/5.0, 0.0, 1-i/5.0, 1.0), 80) for i in range(0, 6)]
primitives += [Point(vec3((i+1)*100 + 100, (i+1)*100, 0), vec4(1-i/5.0, 0.0,   i/5.0, 1.0), 80) for i in range(0, 6)]

canvas = Canvas()
for primitive in primitives:
    canvas << primitive;

while not renderer.should_close():
    canvas.render()
