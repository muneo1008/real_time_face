# 😀실시간 얼굴 탐지 및 효과

---
# 🧑‍💻실행법
### 1. pip install -r requirements.txt


### 2. dnn 모델 관련 파일 다운로드(두 파일 모두 루트 경로에 저장)
* [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
* [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)


### 3. app.py 실행

# 📌 기능
1. Haar 또는 DNN(Caffe 모델) 기반 얼굴 탐지 방법 선택 기능
2. 박스, 모자이크, 블러 중 하나를 얼굴 효과 처리 기능
3. 실시간으로 인식된 얼굴 수 GUI에 표시
4. 민감도 조절 기능
5. 이미지 업로드 및 효과 적용, 저장 기능
6. 웹캠 영상 녹화 및 저장

# 📅개발
### - 프로젝트 시작 ~ 2025.05.07
* 가상환경에서 실행 중 깃과의 알 수 없는 오류로 인해 프로젝트 재공유


* Haar과 DNN 모델을 활용한 얼굼 탐지와 효과 적용 완료

### - 2025.05.11
* 2명 얼굴 인식
* 감지 얼굴 수 추가
* 민감도 수정 바 추가

### - 2025.06.06
* GUI 변경
* 이미지 업로드 및 효과 적용, 저장 기능
* 웹캠 영상 녹화 및 저장

# 🔗참고
* [DNN 참고 링크](https://mslilsunshine.tistory.com/70)
* [Haar Cascade 참고 링크](https://bkshin.tistory.com/entry/%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84-1-%ED%95%98%EB%A5%B4-%EC%BA%90%EC%8A%A4%EC%BC%80%EC%9D%B4%EB%93%9C-%EC%96%BC%EA%B5%B4-%EA%B2%80%EC%B6%9C-Haar-Cascade-Face-Detection)
