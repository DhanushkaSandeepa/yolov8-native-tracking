import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

LINE_START = sv.Point(0, 360)
LINE_END = sv.Point(1280, 360)

def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO("YOUR/MODEL")
    for result in model.track(source="YOUR/VIDEO/PATH", show=True, stream=True, agnostic_nms=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        valid_indices = np.where(detections.class_id == 0)[0]
        valid_detections = detections[valid_indices]

        for detection in valid_detections:
            tracker_id = detection[4]
            class_id = 0  # Assuming the class ID is always 0
            confidence = detection[2]

            labels = [
                f"{tracker_id} {class_id} {confidence:0.2f}"
                for _,_, confidence, class_id, tracker_id
                in detections
            ]
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            line_counter.trigger(detections=detections)  # Assuming bounding box coordinates are at index 2
            line_annotator.annotate(frame=frame, line_counter=line_counter)

            cv2.imshow("yolov8", frame)

            if (cv2.waitKey(30) == 27):
                break


if __name__ == "__main__":
    main()
