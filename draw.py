from tkinter import *
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def draw_image():
	canvas_w, canvas_h = (500,500)
	black = (0,0,0)

	def paint(event):
		x1,y1 = (event.x - 1), (event.y - 1)
		x2,y2 = (event.x + 1), (event.y + 1)
		w.create_oval(x1,y1,x2,y2, fill='white',width=3)
		draw.line([x1, y1, x2, y2],fill="white",width=3)

	def exit():
		root.destroy()

	def main():
		global root
		root = Tk()
		root.title('Drawing Shapes')
		w = Canvas(root,
					width=canvas_w,height=canvas_h,bg='white')
		w.pack(expand= YES, fill = BOTH)
		w.bind("<B1-Motion>",paint)

		return w,root

	w,root = main()

	image1 = Image.new("RGB", (500, 500),black)
	draw = ImageDraw.Draw(image1)

	def buttons():
		button = Button(text='Run',command=exit)
		button.pack()
		return button

	b = buttons()
	root.mainloop()
	return image1

def showImage():
	image1 = draw_image()
	img = np.array(image1)
	img = img.reshape(500,500,3)
	return img