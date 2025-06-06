import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import haar_detector
import dnn_detector

# 초기 설정
cap = cv2.VideoCapture(0)
running = True
file_mode = "webcam"  # webcam, image, video
last_frame = None

# GUI
root = tk.Tk()
root.title("실시간 얼굴 탐지")

method_var = tk.StringVar(value="haar")
effect_var = tk.StringVar(value="box")
confidence_var = tk.DoubleVar(value=0.5)

video_label = tk.Label(root)
video_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

# 탐지 방법 선택
method_frame = ttk.LabelFrame(root, text="탐지 방법")
method_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")
ttk.Radiobutton(method_frame, text="Haar", variable=method_var, value="haar").pack(padx=5, pady=5)
ttk.Radiobutton(method_frame, text="DNN", variable=method_var, value="dnn").pack(padx=5, pady=5)

# 효과 선택
effect_frame = ttk.LabelFrame(root, text="효과")
effect_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")
ttk.Radiobutton(effect_frame, text="박스", variable=effect_var, value="box").pack(padx=5, pady=5)
ttk.Radiobutton(effect_frame, text="모자이크", variable=effect_var, value="mosaic").pack(padx=5, pady=5)
ttk.Radiobutton(effect_frame, text="블러", variable=effect_var, value="blur").pack(padx=5, pady=5)

# 기타 컨트롤
control_frame = ttk.LabelFrame(root, text="기타")
control_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")

face_count_label = ttk.Label(control_frame, text="감지 얼굴 수: 0")
face_count_label.pack(padx=5, pady=5)

ttk.Label(control_frame, text="민감도 (0~1)").pack(padx=5, pady=(0, 2))
confidence_slider = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=confidence_var, orient="horizontal")
confidence_slider.pack(padx=5, pady=5)

# 파일 열기
def open_file():
    global cap, file_mode, running
    filepath = filedialog.askopenfilename(
        filetypes=[("Video/Image Files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png")]
    )
    if filepath:
        cap.release()
        if filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_mode = 'image'
        else:
            file_mode = 'video'
        cap = cv2.VideoCapture(filepath)
        if file_mode == 'image':
            update_frame()  # 이미지 처리는 1회만

# 이미지 저장
def save_frame():
    if last_frame is not None:
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if filepath:
            cv2.imwrite(filepath, last_frame)
            messagebox.showinfo("저장 완료", f"{os.path.basename(filepath)} 로 저장되었습니다.")

ttk.Button(control_frame, text="파일 열기", command=open_file).pack(padx=5, pady=5)
ttk.Button(control_frame, text="이미지 저장", command=save_frame).pack(padx=5, pady=5)

# 종료 함수
def on_close():
    global running
    running = False
    cap.release()
    root.destroy()

ttk.Button(control_frame, text="나가기", command=on_close).pack(padx=5, pady=30)

# 이미지 처리 및 디스플레이 함수
def process_and_display(frame):
    global last_frame
    method = method_var.get()
    effect = effect_var.get()
    confidence = confidence_var.get()

    if method == "haar":
        frame, face_count = haar_detector.detect_faces(frame, effect, confidence)
    else:
        frame, face_count = dnn_detector.detect_faces(frame, effect, confidence)

    face_count_label.config(text=f"감지 얼굴 수: {face_count}")
    last_frame = frame.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# 프레임 업데이트 함수
def update_frame():
    if not running:
        return

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        process_and_display(frame)

        # 이미지 모드는 1회 처리 후 종료
        if file_mode != 'image':
            root.after(10, update_frame)
    else:
        if file_mode == 'video':
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 영상 반복 재생

# 시작
root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()
