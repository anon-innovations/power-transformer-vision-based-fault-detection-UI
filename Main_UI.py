import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import argparse
import numpy as np
import cv2

threshold = 140
area_of_box = 100        # 3000 for img input
min_temp = 60           # in fahrenheit
font_scale_caution = 1   # 2 for img input
font_scale_temp = 0.7    # 1 for img input

root = Tk()  # create root window
root.title("Transformar Fulti Recognition System")  # title of the GUI window
root.maxsize(900, 600)  # specify the max size the window can expand to
root.config(bg="skyblue")  # specify background color

# Create left and right frames
left_frame = Frame(root, width=250, height=400, bg='grey')
left_frame.grid(row=0, column=0, padx=10, pady=5)

right_frame = Frame(root, width=650, height=400, bg='grey')
right_frame.grid(row=0, column=1, padx=10, pady=5)

# Create tool bar frame
tool_bar = Frame(left_frame, width=200, height=200)
tool_bar.grid(row=2, column=0, padx=5, pady=5)

# Create frames and labels in left_frame
lbl_time = tk.Label(tool_bar, text='Time:', padx=5, pady=5,font=('verdana',10))
lbl_alt = tk.Label(tool_bar, text='Altitude:', padx=5, pady=5,font=('verdana',10))
lbl_hum = tk.Label(tool_bar, text='Humidity:', padx=5, pady=5,font=('verdana',10))
lbl_air = tk.Label(tool_bar, text='Air Flow rate:', padx=5, pady=5,font=('verdana',10))
lbl_cap = tk.Label(tool_bar, text='Transformar:', padx=5, pady=5,font=('verdana',10))
lbl_restxt = tk.Label(tool_bar, text='Results:', padx=15, pady=15,font=('verdana',10))
lbl_res = tk.Label(tool_bar, padx=15, pady=15,font=('verdana',16))

entry_pic_path = tk.Entry(tool_bar, font=('verdana',10))

entry_time = tk.Entry(tool_bar, font=('verdana',10))
entry_alt = tk.Entry(tool_bar, font=('verdana',10))
entry_hum = tk.Entry(tool_bar, font=('verdana',10))
entry_air = tk.Entry(tool_bar, font=('verdana',10))
entry_cap = tk.Entry(tool_bar, font=('verdana',10))

btn_browse = tk.Button(tool_bar, text='Select Image',bg='grey',font=('verdana',10))
btn_submit = tk.Button(tool_bar, text='Submit',bg='grey',font=('verdana',10))


def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature (fahrenheit) depending upon the camera hardware
    """
    return pixel_avg / 2.25

def process_frame(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Binary threshold
    _, binary_thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)

    # Image opening: Erosion followed by dilation
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    # Get contours from the image obtained by opening operation
    contours, _ = cv2.findContours(image_opening, 1, 2)

    image_with_rectangles = np.copy(frame)

    for contour in contours:
        # rectangle over each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Pass if the area of rectangle is not large enough
        if (w) * (h) < area_of_box:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < min_temp else (
            255, 255, 127)

        # Callback function if the following condition is true
        if temperature >= min_temp:
            # Call back function here
            cv2.putText(image_with_rectangles, "High temperature detected !!!", (35, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_caution, color, 2, cv2.LINE_AA)

        # Draw rectangles for visualisation
        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, (x, y), (x+w, y+h), color, 2)

        # Write temperature for each rectangle
        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)
        print("{} C".format(temperature))

    return image_with_rectangles

def selectPic():
    global img
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image", filetypes=(("jpg images","*.jpg"),("png images","*.png")))    
    image = cv2.imread(filename)
    img = process_frame(image)
    img = Image.open(str(filename))
    
    img = img.resize((500,500), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    Label(right_frame, image=img).grid(row=0, column=0, padx=5, pady=5)
    entry_pic_path.insert(0, filename)
    
    saved_img = cv2.imread(str(filename ))

btn_browse['command'] = selectPic

def submit():
        temp = 40
        time = float(entry_time.get())
        alt = int(entry_alt.get())
        hum = int(entry_hum.get())
        air = int(entry_air.get())
        cap = int(entry_cap.get())
        
        if temp <= 50 and hum > 40 and air in range (0,5) and alt in range (10,15) and cap == 100:
                
            lbl_res['text'] = ("Object No Fault Detected")
            
        else:
            lbl_res['text'] = ("Object Fault Detected")
            
btn_submit['command'] = submit

entry_pic_path.grid(row=0, column=0, padx=5, pady=5)
btn_browse.grid(row=1, column=0,  padx=5, pady=5)


lbl_time.grid(row=2, column=0)
entry_time.grid(row=3, column=0)
lbl_alt.grid(row=4, column=0)
entry_alt.grid(row=5, column=0)
lbl_hum.grid(row=6, column=0)
entry_hum.grid(row=7, column=0)
lbl_air.grid(row=8, column=0)
entry_air.grid(row=9, column=0)
lbl_cap.grid(row=10, column=0)
entry_cap.grid(row=11, column=0)

lbl_restxt.grid(row=12, column=0)
lbl_res.grid(row=13, column=0)

btn_submit.grid(row=14, column=0, padx=5, pady=5)

root.mainloop()