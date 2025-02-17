import cv2
import mediapipe as mp
import os
import random

# TensorFlow 및 Mediapipe 로그 억제
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from absl import logging
logging.set_verbosity(logging.ERROR)

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 필터 로드 함수: 폴더 내 특정 필터 PNG 파일 불러오기
def load_specific_filters(folder_path, filter_names):
    filters = {}
    for name in filter_names:
        path = os.path.join(folder_path, f"{name}.png")
        if os.path.exists(path):
            filters[name] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not filters:
        raise FileNotFoundError("지정된 필터 이미지를 찾을 수 없습니다.")
    return filters

# 필터를 이미지에 오버레이
def overlay(image, x, y, w, h, overlay_image):
    try:
        overlay_image_resized = cv2.resize(overlay_image, (w, h))
        alpha = overlay_image_resized[:, :, 3] / 255.0  # 투명도 채널
        for c in range(3):  # BGR 채널 블렌딩
            image[y:y+h, x:x+w, c] = (overlay_image_resized[:, :, c] * alpha +
                                      image[y:y+h, x:x+w, c] * (1 - alpha))
    except Exception as e:
        pass

# 머리 및 필터 적용 영역 계산 함수 (필터 크기 조정 포함)
def get_forehead_coords(landmarks, img_width, img_height, scale_factor=3.0):  # 기본값을 3.0으로 설정
    left_forehead = landmarks[10]
    right_forehead = landmarks[454]
    center_forehead = landmarks[151]

    x1, y1 = int(left_forehead.x * img_width), int(left_forehead.y * img_height)
    x2, y2 = int(right_forehead.x * img_width), int(right_forehead.y * img_height)
    xc, yc = int(center_forehead.x * img_width), int(center_forehead.y * img_height)

    # 이마 영역 크기 계산
    width = abs(x2 - x1)
    height = int(width * 0.5)

    # 필터 크기 조정
    width = int(width * scale_factor)
    height = int(height * scale_factor)

    # 이마 중심을 기준으로 영역 조정
    x = xc - width // 2
    y = yc - height
    return x, y, width, height

# 1번과 2번 중 선택
def choose_filter_set():
    print("1번 아니면 2번을 골라주세요:")
    choice = input("입력 (1 또는 2): ").strip()
    if choice == "1":
        return ["game", "ingong", "mobil", "voan"], None  # 1번 선택 시 확률 없음
    elif choice == "2":
        return ["ingong", "spaser"], [0.7, 0.3]  # 2번 선택 시 확률 설정
    else:
        print("잘못된 입력입니다. 프로그램을 종료합니다.")
        exit()

# 메인 실행
def main():
    # 필터 선택
    selected_filter_names, probabilities = choose_filter_set()
    filters = load_specific_filters("images", selected_filter_names)  # 'images' 폴더에서 필터 로드

    # 실행 시 랜덤으로 필터 선택 (확률 포함)
    if probabilities:
        selected_filter_name = random.choices(selected_filter_names, probabilities)[0]
    else:
        selected_filter_name = random.choice(selected_filter_names)
    selected_filter = filters[selected_filter_name]

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or not filters:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    img_height, img_width, _ = frame.shape

                    # 이마 좌표 계산 (필터 크기 약간 줄임)
                    x, y, w, h = get_forehead_coords(face_landmarks.landmark, img_width, img_height, scale_factor=3.0)

                    # 선택된 필터를 적용
                    overlay(frame, x, y, w, h, selected_filter)

            cv2.imshow("Filtered Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
