import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from customtkinter import CTkImage
import haar_detector
import dnn_detector
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
running = True
file_mode = 'camera'
original_image = None
display_image = None
last_frame = None
recording = False
video_writer = None

def ensure_save_folder():
    save_folder = "save"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder

def current_datetime_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 초기
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("실시간 얼굴 탐지")
root.geometry("1000x600")

method_var = ctk.StringVar(value="haar")
effect_var = ctk.StringVar(value="box")
confidence_var = ctk.DoubleVar(value=0.5)

# GUI
video_frame = ctk.CTkFrame(root, width=700, height=580)
video_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)
video_label = ctk.CTkLabel(video_frame, text="")
video_label.pack(fill="both", expand=True)

sidebar = ctk.CTkFrame(root, width=250)
sidebar.pack(side="right", padx=10, pady=10, fill="y")

def create_section(master, title, widgets):
    frame = ctk.CTkFrame(master, corner_radius=10)
    frame.pack(padx=10, pady=10, fill="x")

    title_label = ctk.CTkLabel(frame, text=title, font=("Arial", 16, "bold"))
    title_label.pack(pady=(10, 5))

    for widget in widgets:
        widget.pack(anchor="w", padx=20, pady=2, fill="x")

    return frame

create_section(
    sidebar,
    "탐지 방법",
    [
        ctk.CTkRadioButton(sidebar, text="Haar", variable=method_var, value="haar", command=lambda: on_option_change()),
        ctk.CTkRadioButton(sidebar, text="DNN", variable=method_var, value="dnn", command=lambda: on_option_change())
    ]
)

create_section(
    sidebar,
    "효과",
    [
        ctk.CTkRadioButton(sidebar, text="박스", variable=effect_var, value="box", command=lambda: on_option_change()),
        ctk.CTkRadioButton(sidebar, text="모자이크", variable=effect_var, value="mosaic", command=lambda: on_option_change()),
        ctk.CTkRadioButton(sidebar, text="블러", variable=effect_var, value="blur", command=lambda: on_option_change())
    ]
)

# 민감도
def update_confidence_label(val):
    confidence_value_label.configure(text=f"{float(val):.2f}")
    on_option_change()

slider = ctk.CTkSlider(sidebar, from_=0.0, to=1.0, variable=confidence_var, command=update_confidence_label)
confidence_value_label = ctk.CTkLabel(sidebar, text=f"{confidence_var.get():.2f}")

create_section(
    sidebar,
    "민감도 (0 ~ 1)",
    [slider, confidence_value_label]
)

# 감지 수
face_count_label = ctk.CTkLabel(sidebar, text="감지 얼굴 수: 0", font=("Arial", 14))
face_count_label.pack(pady=10)

button_row = ctk.CTkFrame(sidebar)
button_row.pack(pady=5, padx=20, fill="x")

webcam_button = ctk.CTkButton(button_row, text="📷 웹캠", command=lambda: use_webcam(), width=100)
webcam_button.pack(side="left", expand=True, padx=(0, 5))

image_button = ctk.CTkButton(button_row, text="🖼️ 이미지", command=lambda: open_file(), width=100)
image_button.pack(side="right", expand=True, padx=(5, 0))

record_button = ctk.CTkButton(sidebar, text="🔴 녹화 시작", fg_color="green", command=lambda: toggle_recording())
record_button.pack(pady=5, padx=20, fill="x")

ctk.CTkButton(sidebar, text="🖼️ 이미지 저장", command=lambda: save_image()).pack(pady=5, padx=20, fill="x")

ctk.CTkButton(sidebar, text="❌ 나가기", command=lambda: on_close()).pack(pady=10, padx=20, fill="x")

# 기능
def on_close():
    global running, video_writer
    running = False
    cap.release()
    if video_writer:
        video_writer.release()
    root.destroy()

def use_webcam():
    global file_mode, cap, original_image, display_image, last_frame
    if cap.isOpened():
        cap.release()
    cap = cv2.VideoCapture(0)
    file_mode = 'camera'
    original_image = None
    display_image = None
    last_frame = None

def open_file():
    global cap, file_mode, original_image, display_image, last_frame
    filepath = filedialog.askopenfilename(filetypes=[("영상/이미지", "*.jpg *.jpeg *.png")])
    if filepath:
        if cap.isOpened():
            cap.release()
        file_mode = 'image'
        img = cv2.imread(filepath)
        if img is not None:
            original_image = img.copy()
            display_image = img.copy()
            last_frame = None
            process_and_display(display_image.copy())

def save_image():
    global last_frame
    if last_frame is not None:
        save_folder = ensure_save_folder()
        filename = f"{current_datetime_str()}.jpg"
        full_path = os.path.join(save_folder, filename)
        cv2.imwrite(full_path, last_frame)
        print(f"이미지 저장 완료: {full_path}")

def on_option_change():
    global display_image, last_frame
    if file_mode == 'image' and original_image is not None:
        display_image = original_image.copy()
        process_and_display(display_image.copy())

def toggle_recording():
    global recording, video_writer, record_button

    if recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        record_button.configure(text="🔴 녹화 시작", fg_color="green")
    else:
        save_folder = ensure_save_folder()
        filename = f"{current_datetime_str()}.mp4"
        full_path = os.path.join(save_folder, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(full_path, fourcc, 20.0, (640, 480))
        recording = True
        record_button.configure(text="⏹ 녹화 중지", fg_color="red")
        print(f"녹화 시작: {full_path}")

def process_and_display(frame):
    global last_frame, recording, video_writer

    method = method_var.get()
    effect = effect_var.get()
    confidence = confidence_var.get()

    if method == "haar":
        frame, face_count = haar_detector.detect_faces(frame, effect, confidence)
    else:
        frame, face_count = dnn_detector.detect_faces(frame, effect, confidence)

    face_count_label.configure(text=f"감지 얼굴 수: {face_count}")
    last_frame = frame.copy()

    if recording and video_writer:
        resized_frame = cv2.resize(frame, (640, 480))
        video_writer.write(resized_frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    ctk_img = CTkImage(light_image=img, size=(video_label.winfo_width(), video_label.winfo_height()))
    video_label.configure(image=ctk_img)
    video_label.imgtk = ctk_img

def update_frame():
    global running
    if not running:
        return

    if file_mode == 'camera':
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            process_and_display(frame)

    root.after(10, update_frame)

# 실행
root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()
