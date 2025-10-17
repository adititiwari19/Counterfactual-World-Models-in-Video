import torch
import open_clip
import whisper
from pathlib import Path
import cv2
from time import sleep

video_path = Path(__file__).parent / "how_to_tie_a_tie.mp4"

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Video not found."
delay_ms = int(1000 / cap.get(cv2.CAP_PROP_FPS))


def runVideo(wait):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    while ret:
        cv2.imshow("Video", frame)

        yield(frame)

        if wait and cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    runVideo()

