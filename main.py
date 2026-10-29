import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque
import os
from gtts import gTTS
import subprocess

# ملف تسجيل الأحداث
match_txt_path = r'output/match_ev.txt'

log_file = open(match_txt_path , "w", encoding='utf-8')
ball_positions = deque(maxlen=30)

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image

    Args:
        img: np.array object of shape (WxHx3) that represents the BGR value of the
        frame pixels .

    Returns:
        grass_color
            Tuple of the BGR value of the grass color in the image
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
  """
  Finds the images of the players in the frame and their bounding boxes.

  Args:
      result: ultralytics.engine.results.Results object that contains all the
      result of running the object detection algroithm on the frame

  Returns:
      players_imgs
          List of np.array objects that contain the BGR values of the cropped
          parts of the image that contains players.
      players_boxes
          List of ultralytics.engine.results.Boxes objects that contain various
          information about the bounding boxes of the players found in the image.
  """
  players_imgs = []
  players_boxes = []
  for box in result.boxes:
    label = int(box.cls.numpy()[0])
    if label == 0:
      x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
      player_img = result.orig_img[y1: y2, x1: x2]
      players_imgs.append(player_img)
      players_boxes.append(box)
  return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
  """
  Finds the kit colors of all the players in the current frame

  Args:
      players: List of np.array objects that contain the BGR values of the image
      portions that contain players.
      grass_hsv: tuple that contain the HSV color value of the grass color of
      the image background.

  Returns:
      kits_colors
          List of np arrays that contain the BGR values of the kits color of all
          the players in the current frame
  """
  kits_colors = []
  if grass_hsv is None:
    grass_color = get_grass_color(frame)
    grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

  for player_img in players:
      # Convert image to HSV color space
      hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

      # Define range of green color in HSV
      lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
      upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

      # Threshold the HSV image to get only green colors
      mask = cv2.inRange(hsv, lower_green, upper_green)

      # Bitwise-AND mask and original image
      mask = cv2.bitwise_not(mask)
      upper_mask = np.zeros(player_img.shape[:2], np.uint8)
      upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
      mask = cv2.bitwise_and(mask, upper_mask)

      kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

      kits_colors.append(kit_color)
  return kits_colors

def get_kits_classifier(kits_colors):
  """
  Creates a K-Means classifier that can classify the kits accroding to their BGR
  values into 2 different clusters each of them represents one of the teams

  Args:
      kits_colors: List of np.array objects that contain the BGR values of
      the colors of the kits of the players found in the current frame.

  Returns:
      kits_kmeans
          sklearn.cluster.KMeans object that can classify the players kits into
          2 teams according to their color..
  """
  kits_kmeans = KMeans(n_clusters=2)
  kits_kmeans.fit(kits_colors);
  return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
  """
  Classifies the player into one of the two teams according to the player's kit
  color

  Args:
      kits_classifer: sklearn.cluster.KMeans object that can classify the
      players kits into 2 teams according to their color.
      kits_colors: List of np.array objects that contain the BGR values of
      the colors of the kits of the players found in the current frame.

  Returns:
      team
          np.array object containing a single integer that carries the player's
          team number (0 or 1)
  """
  team = kits_classifer.predict(kits_colors)
  return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
  """
  Finds the label of the team that is on the left of the screen

  Args:
      players_boxes: List of ultralytics.engine.results.Boxes objects that
      contain various information about the bounding boxes of the players found
      in the image.
      kits_colors: List of np.array objects that contain the BGR values of
      the colors of the kits of the players found in the current frame.
      kits_clf: sklearn.cluster.KMeans object that can classify the players kits
      into 2 teams according to their color.
  Returns:
      left_team_label
          Int that holds the number of the team that's on the left of the image
          either (0 or 1)
  """
  left_team_label = 0
  team_0 = []
  team_1 = []

  for i in range(len(players_boxes)):
    x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

    team = classify_kits(kits_clf, [kits_colors[i]]).item()
    if team==0:
      team_0.append(np.array([x1]))
    else:
      team_1.append(np.array([x1]))

  team_0 = np.array(team_0)
  team_1 = np.array(team_1)

  if np.average(team_0) - np.average(team_1) > 0:
    left_team_label = 1

  return left_team_label

def detect_goal(ball_x, ball_y, frame_width, frame_height):
    # مرمى الجهة اليسرى
    left_goal_x_min = 0
    left_goal_x_max = int(frame_width * 0.15)
    goal_y_min = int(frame_height * 0.3)
    goal_y_max = int(frame_height * 0.7)

    # مرمى الجهة اليمنى
    right_goal_x_min = int(frame_width * 0.85)
    right_goal_x_max = frame_width

    # تحقق إذا كانت الكرة في أي من المرميين
    in_left_goal = left_goal_x_min <= ball_x <= left_goal_x_max and goal_y_min <= ball_y <= goal_y_max
    in_right_goal = right_goal_x_min <= ball_x <= right_goal_x_max and goal_y_min <= ball_y <= goal_y_max

    return in_left_goal or in_right_goal





def annotate_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_output_path = 'output/final_with_audio.mp4'
    output_video_path = 'output/newvideos.mp4'

    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    kits_clf = None
    left_team_label = 0
    grass_hsv = None
    
    if not hasattr(annotate_video, 'last_player'):
        annotate_video.last_player = None

    while cap.isOpened():
        success, frame = cap.read()
        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not success:
            break

        annotated_frame = cv2.resize(frame, (width, height))
        result = model(annotated_frame, conf=0.5, verbose=False)[0]

        players_imgs, players_boxes = get_players_boxes(result)
        kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

        if current_frame_idx == 1:
            kits_clf = get_kits_classifier(kits_colors)
            left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
            grass_color = get_grass_color(result.orig_img)
            grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

        ball_x, ball_y = None, None

        for box in result.boxes:
            label = int(box.cls.numpy()[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())

            if label == 0:
                kit_color = get_kits_colors([result.orig_img[y1: y2, x1: x2]], grass_hsv)
                team = classify_kits(kits_clf, kit_color)
                label = 0 if team == left_team_label else 1

            elif label == 1:
                label = 2 if x1 < 0.5 * width else 3

            elif label == 4:
                ball_x, ball_y = (x1 + x2) // 2, (y1 + y2) // 2
                ball_positions.append((ball_x, ball_y))

                min_dist = float("inf")
                closest_player = None
                for i, p_box in enumerate(players_boxes):
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0].numpy())
                    player_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                    dist = np.linalg.norm(np.array(player_center) - np.array((ball_x, ball_y)))
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = i


                if detect_goal(ball_x, ball_y, width, height):
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        with open(match_txt_path, "a", encoding='utf-8') as log_file:
                            log_file.write(f"⚽️ هدف في الدقيقة {timestamp:.2f}\n")


                if closest_player is not None:
                    current_player = f"Player_{closest_player}"
                    if annotate_video.last_player and annotate_video.last_player != current_player:
                        with open(match_txt_path, "a", encoding='utf-8') as log_file:
                            log_file.write(f"{annotate_video.last_player} مرر الكرة إلى {current_player} في الوقت {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f} ثانية\n")
                    annotate_video.last_player = current_player

            elif label == 8 and ball_x and ball_y:
                gx1, gy1, gx2, gy2 = map(int, box.xyxy[0].numpy())
                if gx1 < ball_x < gx2 and gy1 < ball_y < gy2:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    log_file.write(f"⚽️ هدف من {annotate_video.last_player} عند الدقيقة {timestamp:.2f}\n")

            label += 2 if label < 4 else 0
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
            cv2.putText(annotated_frame, labels[label], (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        box_colors[str(label)], 2)

        output_video.write(annotated_frame)

    log_file.close()
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    print("🔊 جاري تحويل النص إلى صوت...")
    with open(match_txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tts = gTTS(text, lang='ar')
    audio_path = "output/commentary.mp3"
    tts.save(audio_path)
    print("✅ تم حفظ التعليق الصوتي.")
    print("🎬 جاري دمج الصوت مع الفيديو باستخدام FFmpeg...")
    command = [
        'ffmpeg',
        '-y',  # للكتابة فوق الملف إذا كان موجود
        '-i', output_video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        final_output_path
    ]

    subprocess.run(command)

    print(f"✅ تم حفظ الفيديو النهائي مع الصوت في: {final_output_path}")


if __name__ == "__main__":
    labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff", "Goal"]
    box_colors = {
        "0": (150, 50, 50),
        "1": (37, 47, 150),
        "2": (41, 248, 165),
        "3": (166, 196, 10),
        "4": (155, 62, 157),
        "5": (123, 174, 213),
        "6": (217, 89, 204),
        "7": (22, 11, 15),
        "8": (0, 255, 255)
    }

    model = YOLO(r"D:\sport\Football-Object-Detection-main\Football-Object-Detection-main\weights\last.pt")
    video_path = r"D:\sport\Football-Object-Detection-main\Football-Object-Detection-main\test_videos\pass.mp4"


    annotate_video(video_path,model)