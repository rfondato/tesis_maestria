import cv2
from preprocess import Preprocessor
from display import add_contour_keypoints
from har_model import HARModule
import numpy as np
import torch as t
from constants import LABELS

cap = cv2.VideoCapture(0)
preprocessor = Preprocessor(resolution=100, seq_length=30)

model = HARModule.load_from_checkpoint("./models/CNN-Res100-Seq30-epoch=29-val_loss=0.01.ckpt", device="cuda")
model.eval()

def predict(frame, features):
    frame = cv2.resize(frame, (640, 480))
    coords, center = add_contour_keypoints(frame, features[-1])

    label_text = "UNDEFINED"

    if (len(features) == 30):
        input = t.from_numpy(np.expand_dims(np.expand_dims(np.array(features, np.float32), axis=0), axis=0)).to("cuda")
        pred = model(input)
        probs = t.nn.Softmax()(pred)
        pred_proba, pred = t.max(probs, dim=1)
        label_text = list(LABELS.keys())[list(LABELS.values()).index(int(pred[0]))].upper() if float(pred_proba[0]) > 0.7 else "UNDEFINED"
               
    #top_left_corner = coords.min(axis=0)
    cv2.putText(frame, label_text, (int(center[1]), int(center[0])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2);

    return frame

while cap.isOpened():
    success, frame = cap.read()

    if success:
        features = preprocessor.process_frame(frame)

        if features is not None:
            frame = predict(frame, features)
                
        cv2.imshow("Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
