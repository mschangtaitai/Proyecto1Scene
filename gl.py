# Michael Chan 18562 
# Graficas por Computadora 
# gl

import struct
import random
from obj import Obj, Texture
from collections import namedtuple
from numpy import matrix, cos, sin, tan

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

def char(input):
	return struct.pack('=c', input.encode('ascii'))

def word(input):
	return struct.pack('=h', input)

def dword(input):
	return struct.pack('=l', input)

def glColor(r, g, b):
	return bytes([b, g, r])

def cross(v1, v2):
	return V3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x,
	)

def bbox(*vertices):
	xs = [ vertex.x for vertex in vertices]
	ys = [ vertex.y for vertex in vertices]

	xs.sort()
	ys.sort()


	xmin = int(xs[0])
	ymin = int(ys[0])
	xmax = int(xs[-1])
	ymax = int(ys[-1])

	return xmin, xmax, ymin, ymax

def barycentric(A, B, C, P):
	cx, cy, cz = cross(
		V3(C.x - A.x, B.x - A.x, A.x - P.x), 
		V3(C.y - A.y, B.y - A.y, A.y - P.y)
	)

	if abs(cz) < 1:
		return -1, -1, -1

	u = cx/cz
	v = cy/cz
	w = 1 - (u + v)

	return V3(w, v, u)


def sum(v0, v1):
	return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
	return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
	return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
	return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def cross(v0, v1):
	return V3(
		v0.y * v1.z - v0.z * v1.y,
		v0.z * v1.x - v0.x * v1.z,
		v0.x * v1.y - v0.y * v1.x,
	)

def length(v0):
	return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):

	v0length = length(v0)

	if not v0length:
		return V3(0, 0, 0)

	return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

BLACK = glColor(0, 0, 0)
WHITE = glColor(255, 255, 255)


class Render(object):
	def glInit(self, width, height):
		self.width = width
		self.height = height
		self.color = glColor(255, 255, 255)
		self.clearColor = glColor(0, 0, 0)
		self.light = V3(0,0,1)
		self.activeVertexArray = []
		self.glClear()

		self.zbuffer = [
			[-float('inf') for x in range(self.width)]
			for y in range(self.height)
		]

	def glColorPoint(self, r, g, b):
		self.color = glColor(round(r * 255), round(g * 255), round(b * 255))

	def glCreateWindow(self, width = 640, height = 480):
		self.width = width
		self.height = height

	def glClear(self):
		self.framebuffer = [
			[glColor(0,0,80) for i in range(self.width)]
			for j in range(self.height)
		]

		self.zbuffer = [
			[-float('inf') for x in range(self.width)]
			for y in range(self.height)
		]

	def glClearColor(self, r, g, b):
		self.clearColor = glColor(round(r * 255), round(g * 255), round(b * 255))
		self.framebuffer = [
            [clearColor for x in range(self.width)] for y in range(self.height)
        ]

	def pixel(self, x, y, color):
		try:
			self.framebuffer[y%self.height][x%self.width] = color
		except:
			pass

	def glFinish(self, filename):
		f = open(filename, 'bw')

		f.write(char('B'))
		f.write(char('M'))
		f.write(dword(54 + self.width * self.height * 3))
		f.write(dword(0))
		f.write(dword(54))

		f.write(dword(40))
		f.write(dword(self.width))
		f.write(dword(self.height))
		f.write(word(1))
		f.write(word(24))
		f.write(dword(0))
		f.write(dword(self.width * self.height * 3))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))

		for x in range(self.height):
			for y in range(self.width):
				# print(x, y)
				f.write(self.framebuffer[x][y])

		f.close()

	def glLine(self, A, B, color):

		x1 = round(A.x)
		y1 = round(A.y)
		x2 = round(B.x)
		y2 = round(B.y)

		dy = abs(y2 - y1)
		dx = abs(x2 - x1)
		steep = dy > dx

		if steep:
		    x1, y1 = y1, x1
		    x2, y2 = y2, x2

		if x1 > x2:
		    x1, x2 = x2, x1
		    y1, y2 = y2, y1

		dy = abs(y2 - y1)
		dx = abs(x2 - x1)

		offset = 0
		threshold = dx

		y = y1
		for x in range(x1, x2 + 1):
		    if steep:
		        self.pixel(y, x, color)
		    else:
		        self.pixel(x, y, color)
		    
		    offset += dy * 2
		    if offset >= threshold:
		        y += 1 if y1 < y2 else -1
		        threshold += dx * 2

	def transform(self, vertex):
		augmented_vertex = [
			vertex.x,
			vertex.y,
			vertex.z,
			1
		]
		tranformed_vertex = self.Viewport @ self.Projection @ self.View @ self.Model @ augmented_vertex

		tranformed_vertex = tranformed_vertex.tolist()[0]

		tranformed_vertex = [
			(tranformed_vertex[0]/tranformed_vertex[3]),
			(tranformed_vertex[1]/tranformed_vertex[3]),
			(tranformed_vertex[2]/tranformed_vertex[3])
		]
		# print(V3(*tranformed_vertex))
		return V3(*tranformed_vertex)
	    

	def drawArrays(self, polygonType):
		if polygonType == 'TRIANGLE':
			try:
				cont = 1
				while True:
					self.triangle()
					print("triangulo: ",cont)
					cont += 1
			except StopIteration:
				print("Render done")
		elif polygonType == 'WIREFRAME':
			try:
				cont = 1
				while True:
					self.triangleWireframe()
					print("triangulo: ",cont)
					cont += 1
			except StopIteration:
				print("Render done")
		elif polygonType == 'SHADER':
			try:
				cont = 1
				while True:
					self.triangleShader()
					print("triangulo: ",cont)
					cont += 1
			except StopIteration:
				print("Render done")
		elif polygonType == 'GREYSHADER':
			try:
				cont = 1
				while True:
					self.triangleGreyShade()
					print("triangulo: ",cont)
					cont += 1
			except StopIteration:
				print("Render done")

	def load(self, filename, translate =(0,0,0), scale=(1,1,1), rotate=(0,0,0)):
		self.loadModelMatrix(translate, scale, rotate)
		model = Obj(filename)

		vertexBufferObject = []

		for face in model.faces:

			for facepart in face:

				vertex = self.transform(V3(*model.vertices[facepart[0]-1]))
				vertexBufferObject.append(vertex)

				try:
					tvertex = V3(*model.tvertices[facepart[1]-1])
				except:
					tvertex = V3(*model.tvertices[facepart[1]-1],0)

				# print(tvertex)
				vertexBufferObject.append(tvertex)

		print("ya sali")
		self.activeVertexArray = iter(vertexBufferObject)

	def loadModelMatrix(self, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0)):
		translate = V3(*translate)
		scale = V3(*scale)
		rotate = V3(*rotate)

		translation_matrix = matrix([
			[1, 0, 0, translate.x],
			[0, 1, 0, translate.y],
			[0, 0, 1, translate.z],
			[0, 0, 0, 1],
		])


		a = rotate.x
		rotation_matrix_x = matrix([
			[1, 0, 0, 0],
			[0, cos(a), -sin(a), 0],
			[0, sin(a),  cos(a), 0],
			[0, 0, 0, 1]
		])

		a = rotate.y
		rotation_matrix_y = matrix([
			[cos(a), 0,  sin(a), 0],
			[     0, 1,       0, 0],
			[-sin(a), 0,  cos(a), 0],
			[     0, 0,       0, 1]
		])

		a = rotate.z
		rotation_matrix_z = matrix([
			[cos(a), -sin(a), 0, 0],
			[sin(a),  cos(a), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		])

		rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

		scale_matrix = matrix([
			[scale.x, 0, 0, 0],
			[0, scale.y, 0, 0],
			[0, 0, scale.z, 0],
			[0, 0, 0, 1],
		])

		self.Model = translation_matrix @ rotation_matrix @ scale_matrix

	def loadViewMatrix(self, x, y, z, center):
		M = matrix([
			[x.x, x.y, x.z,  0],
			[y.x, y.y, y.z, 0],
			[z.x, z.y, z.z, 0],
			[0,     0,   0, 1]
		])

		O = matrix([
			[1, 0, 0, -center.x],
			[0, 1, 0, -center.y],
			[0, 0, 1, -center.z],
			[0, 0, 0, 1]
		])

		self.View = M @ O

	def loadProjectionMatrix(self, coeff):
		self.Projection =  matrix([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, coeff, 1]
		])

	def loadViewportMatrix(self, x = 0, y = 0):
		self.Viewport =  matrix([
			[self.width/2, 0, 0, x + self.width/2],
			[0, self.height/2, 0, y + self.height/2],
			[0, 0, 128, 128],
			[0, 0, 0, 1]
		])

	def lookAt(self, eye, center, up):
		z = norm(sub(eye, center))
		x = norm(cross(up, z))
		y = norm(cross(z, x))
		self.loadViewMatrix(x, y, z, center)
		self.loadProjectionMatrix(-1 / length(sub(eye, center)))
		self.loadViewportMatrix()

	def shader(self, x, y):

		xLevel = x/self.width
		return glColor(255,255,255)	

	def triangleShader(self, vertices=(), tvertices=(), texture=None):
		A = next(self.activeVertexArray)
		tA = next(self.activeVertexArray)
		B = next(self.activeVertexArray)
		tB = next(self.activeVertexArray)
		C = next(self.activeVertexArray)
		tC = next(self.activeVertexArray)

		xmin, xmax, ymin, ymax = bbox(A, B, C)

		for x in range(xmin, xmax + 1):
			for y in range(ymin, ymax + 1):
				P = V2(x, y)
				w, v, u = barycentric(A, B, C, P)
				if w < 0 or v < 0 or u < 0:
					#el punto esta afuera
					continue

				z = A.z * w + B.z * v + C.z * u

				tx = tA.x * w + tB.x * v + tC.x * u
				ty = tA.y * w + tB.y * v + tC.y * u
				# print (tx, ty)

				# tx = 0.1
				# ty = 0.2

				color = self.shader(x, y)

				if z > self.zbuffer[x][y]:
					self.pixel(x, y, color)
					self.zbuffer[x][y] = z
		
	def triangleWireframe(self):
		A = next(self.activeVertexArray)
		tA = next(self.activeVertexArray)
		B = next(self.activeVertexArray)
		tB = next(self.activeVertexArray)
		C = next(self.activeVertexArray)
		tC = next(self.activeVertexArray)

		self.glLine(A,B,self.color)
		self.glLine(B,C,self.color)
		self.glLine(C,A,self.color)

	def triangle(self):
		A = next(self.activeVertexArray)
		tA = next(self.activeVertexArray)
		B = next(self.activeVertexArray)
		tB = next(self.activeVertexArray)
		C = next(self.activeVertexArray)
		tC = next(self.activeVertexArray)

		xmin, xmax, ymin, ymax = bbox(A, B, C)
		# print("xmin",xmin)
		# print("xmax",xmax)
		# print("ymin",ymin)
		# print("ymax",ymax)
		# print("___________")
		# print("zburffer: ", len(self.zbuffer), len(self.zbuffer[0]))
		normal = norm(cross(sub(B, A), sub(C, A)))
		intensity = dot(normal, self.light)
		if intensity < 0:
			return 0

		for x in range(xmin, xmax + 1):
			for y in range(ymin, ymax + 1):
				P = V2(x, y)
				w, v, u = barycentric(A, B, C, P)
				if w < 0 or v < 0 or u < 0:
					#el punto esta afuera
					continue

				z = A.z * w + B.z * v + C.z * u

				tx = tA.x * w + tB.x * v + tC.x * u
				ty = tA.y * w + tB.y * v + tC.y * u
				# print (tx, ty)

				color = self.texture.getColor(tx, ty)


				if z > self.zbuffer[x%self.width][y%self.height]:
					self.pixel(x, y, color)
					self.zbuffer[x%self.width][y%self.height] = z

	def triangleGreyShade(self):
		A = next(self.activeVertexArray)
		tA = next(self.activeVertexArray)
		B = next(self.activeVertexArray)
		tB = next(self.activeVertexArray)
		C = next(self.activeVertexArray)
		tC = next(self.activeVertexArray)

		xmin, xmax, ymin, ymax = bbox(A, B, C)
		# print("xmin",xmin)
		# print("xmax",xmax)
		# print("ymin",ymin)
		# print("ymax",ymax)
		# print("___________")
		# print("zburffer: ", len(self.zbuffer), len(self.zbuffer[0]))
		normal = norm(cross(sub(B, A), sub(C, A)))
		intensity = dot(normal, self.light)
		grey = round((intensity * 255)/20)
		if intensity < 0:
			return 0

		for x in range(xmin, xmax + 1):
			for y in range(ymin, ymax + 1):
				P = V2(x, y)
				w, v, u = barycentric(A, B, C, P)
				if w < 0 or v < 0 or u < 0:
					#el punto esta afuera
					continue

				z = A.z * w + B.z * v + C.z * u

				color = glColor(grey,grey,grey)


				if z > self.zbuffer[x%self.width][y%self.height]:
					self.pixel(x, y, color)
					self.zbuffer[x%self.width][y%self.height] = z
