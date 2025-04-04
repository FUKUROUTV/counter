import cv2 as cv
import os
import time
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "xcb"  # X11用の環境変数設定

# 🔹 カスケード分類器の読み込み
custom_cascade = cv.CascadeClassifier('cascade.xml')

# 🔹 カメラを起動
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 🔹 保存フォルダの作成
save_folder = "./captured_images/"
os.makedirs(save_folder, exist_ok=True)
log_file_path = os.path.join(save_folder, "entry_exit_log.txt")

max_images = 500  # 🔹 保存する最大画像数

frame_count = 0
inside_count = 0  # 現在室内にいる人数
entry_count = 0   # 入室した回数
exit_count = 0    # 退室した回数
tracking_objects = {}  # 🔹 追跡中のオブジェクト {ID: 座標リスト}
next_id = 0  # 🔹 新しいオブジェクトに割り当てる ID
tracking_started = False  # 🔹 追跡が開始されたかどうか

max_history = 50  # 🔹 保存する座標の最大数（軌跡の長さ）
exit_threshold = 30  # 🔹 画面端からこの距離以内で消えたら「外に行った」と判定
frame_width = int(cap.get(3))  # 画面の幅
frame_height = int(cap.get(4))  # 画面の高さ
distance_threshold = 60  # 🔹 1フレーム内での移動距離の閾値

def manage_saved_images():
    images = sorted([f for f in os.listdir(save_folder) if f.endswith(".jpg")])
    while len(images) > max_images:
        os.remove(os.path.join(save_folder, images.pop(0)))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 🔹 物体検出
    custom_rects = custom_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.07,
        minNeighbors=2,
        minSize=(100, 100),
        maxSize=(150, 150)
    )

    current_positions = []  # 🔹 現在フレームのオブジェクト位置

    for (x, y, w, h) in custom_rects:
        center_x = x + w // 2
        center_y = y + h // 2
        current_positions.append((center_x, center_y))

    # 🔹 追跡開始の判定（最初にオブジェクトが検出されるまで待機）
    if not tracking_started and len(current_positions) > 0:
        tracking_started = True  # 追跡開始
        for pos in current_positions:
            tracking_objects[next_id] = [pos]  # 初期座標を記録
            next_id += 1

    elif tracking_started:
        # 🔹 既存オブジェクトと新オブジェクトを紐づける（最近傍探索）
        updated_objects = {}  # 更新後のオブジェクトデータ

        for obj_id, past_positions in tracking_objects.items():
            last_position = past_positions[-1]  # オブジェクトの最後の座標

            # 🔹 一番近い新しいオブジェクトを探す（閾値以内でのみ識別）
            min_distance = float('inf')
            nearest_pos = None

            for pos in current_positions:
                dist = np.linalg.norm(np.array(last_position) - np.array(pos))  # ユークリッド距離

                # 🔹 距離が最小かつ閾値内の場合のみ更新対象とする
                if dist < min_distance and dist <= distance_threshold:
                    min_distance = dist
                    nearest_pos = pos

            if nearest_pos:
                # 🔹 追跡オブジェクトを更新
                updated_objects[obj_id] = past_positions + [nearest_pos]
                current_positions.remove(nearest_pos)  # マッチした座標は削除

        # 🔹 閾値内に一致しなかった座標は新規オブジェクトとして登録
        for pos in current_positions:
            updated_objects[next_id] = [pos]
            next_id += 1

        # 🔹 追跡オブジェクトを更新
        tracking_objects = updated_objects

        # 🔹 画面外に出たオブジェクトの削除
        remove_ids = []
        for obj_id, past_positions in tracking_objects.items():
            last_x, last_y = past_positions[-1]

            # 画面端付近で消えた場合、追跡終了
            if (last_x < exit_threshold or last_x > frame_width - exit_threshold or
                last_y < exit_threshold or last_y > frame_height - exit_threshold):
                remove_ids.append(obj_id)

        for obj_id in remove_ids:
            del tracking_objects[obj_id]

    # 🔹 y座標の閾値を跨いだ場合の入退室カウント
    threshold_y = 278  # 閾値のy座標を変数化
    for obj_id, past_positions in tracking_objects.items():
        if len(past_positions) > 1:
            y_prev = past_positions[-2][1]  # 前の座標のy値
            y_last = past_positions[-1][1]  # 最新の座標のy値

            if y_prev <= threshold_y < y_last:  # 上から下へ（入室）
                # 入室時の画像保存
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = f"{save_folder}entry_{timestamp}.jpg"
                cv.imwrite(img_path, frame)
                manage_saved_images()  # 🔹 画像管理を実行
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{timestamp} - Entry\n")
                inside_count -= 1
                entry_count += 1
                print(f"Entry Count: {entry_count}, Exit Count: {exit_count}, Inside Count: {inside_count}")
            elif y_prev > threshold_y >= y_last:  # 下から上へ（退室）
                # 退室時の画像保存
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = f"{save_folder}exit_{timestamp}.jpg"
                cv.imwrite(img_path, frame)
                manage_saved_images()  # 🔹 画像管理を実行
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{timestamp} - Exit\n")
                inside_count += 1
                exit_count += 1
                print(f"Entry Count: {entry_count}, Exit Count: {exit_count}, Inside Count: {inside_count}")
    # 🔹 軌跡を描画
    for obj_id, past_positions in tracking_objects.items():
        for i in range(1, len(past_positions)):
            cv.line(frame, past_positions[i - 1], past_positions[i], (0, 255, 0), 2)

    # 🔹 検出されたオブジェクトに枠を描画
    for (x, y, w, h) in custom_rects:
        if y > 300:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), thickness=3)
        else:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

    # 🔹 左上に文字と数字を表示（例：「Tracking Objects: 数値」）
    display_text = f"Tracking Objects: {len(tracking_objects)} | Inside: {inside_count}"
    cv.putText(frame, display_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # 🔹 カメラ映像を表示
    cv.imshow('Object Tracking', frame)

    # 🔹 0.1秒待機
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 🔹 リソースを解放
cap.release()
cv.destroyAllWindows()