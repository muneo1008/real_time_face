import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import haar_detector
import dnn_detector

#초기
cap = cv2.VideoCapture(0)
running = True



#GUI
root = tk.Tk()
root.title("Face Detection App")

method_var = tk.StringVar(value="haar")
effect_var = tk.StringVar(value="box")

video_label = tk.Label(root)
video_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

method_frame = ttk.LabelFrame(root, text="Detection Method")
method_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")

ttk.Radiobutton(method_frame, text="Haar", variable=method_var, value="haar").pack(padx=5, pady=5)
ttk.Radiobutton(method_frame, text="DNN", variable=method_var, value="dnn").pack(padx=5, pady=5)

effect_frame = ttk.LabelFrame(root, text="Effect")
effect_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

ttk.Radiobutton(effect_frame, text="Box", variable=effect_var, value="box").pack(padx=5, pady=5)
ttk.Radiobutton(effect_frame, text="Mosaic", variable=effect_var, value="mosaic").pack(padx=5, pady=5)
ttk.Radiobutton(effect_frame, text="Blur", variable=effect_var, value="blur").pack(padx=5, pady=5)

control_frame = ttk.LabelFrame(root, text="Control")
control_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")

#종료
def on_close():
    global running
    running = False
    cap.release()
    root.destroy()

ttk.Button(control_frame, text="Quit", command=on_close).pack(padx=5, pady=30)

#영상 업데이트
def update_frame():
    if not running:
        return

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)

        method = method_var.get()
        effect = effect_var.get()

        if method == "haar":
            frame = haar_detector.detect_faces(frame, effect)
        else:
            frame = dnn_detector.detect_faces(frame, effect)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10, update_frame)


root.protocol("WM_DELETE_WINDOW", on_close)

update_frame()
root.mainloop()
