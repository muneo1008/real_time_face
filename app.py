import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from customtkinter import CTkImage
import haar_detector
import dnn_detector

cap = cv2.VideoCapture(0)
running = True
file_mode = 'camera'
original_image = None
display_image = None
last_frame = None

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

face_count_label = ctk.CTkLabel(sidebar, text="감지 얼굴 수: 0", font=("Arial", 14))
face_count_label.pack(pady=10)

ctk.CTkButton(sidebar, text="파일 열기", command=lambda: open_file()).pack(pady=5, padx=20, fill="x")
ctk.CTkButton(sidebar, text="이미지 저장", command=lambda: save_image()).pack(pady=5, padx=20, fill="x")
ctk.CTkButton(sidebar, text="나가기", command=lambda: on_close()).pack(pady=10, padx=20, fill="x")

# 기능
def on_close():
    global running
    running = False
    cap.release()
    root.destroy()

def open_file():
    global cap, file_mode, original_image, display_image, last_frame
    filepath = filedialog.askopenfilename(filetypes=[("영상/이미지", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png")])
    if filepath:
        cap.release()
        if filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_mode = 'image'
            img = cv2.imread(filepath)
            if img is not None:
                original_image = img.copy()
                display_image = img.copy()
                last_frame = None
                process_and_display(display_image.copy())
        else:
            file_mode = 'video'
            cap = cv2.VideoCapture(filepath)
            update_frame()

def save_image():
    global last_frame
    if last_frame is not None:
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, last_frame)

def on_option_change():
    global display_image, last_frame
    if file_mode == 'image' and original_image is not None:
        display_image = original_image.copy()
        process_and_display(display_image.copy())

def process_and_display(frame):
    global last_frame
    method = method_var.get()
    effect = effect_var.get()
    confidence = confidence_var.get()

    if method == "haar":
        frame, face_count = haar_detector.detect_faces(frame, effect, confidence)
    else:
        frame, face_count = dnn_detector.detect_faces(frame, effect, confidence)

    face_count_label.configure(text=f"감지 얼굴 수: {face_count}")
    last_frame = frame.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    ctk_img = CTkImage(light_image=img, size=(video_label.winfo_width(), video_label.winfo_height()))
    video_label.configure(image=ctk_img)
    video_label.imgtk = ctk_img

def update_frame():
    global running
    if not running:
        return

    if file_mode in ['camera', 'video']:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            process_and_display(frame)

    root.after(10, update_frame)

# 실행
root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()
