import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Class labels (same order as training folders)
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DELETE','NOTHING','SPACE'
]

print("Controls: press 'y' to mark prediction as correct, 'n' to mark wrong, 'q' to quit.")

# Start webcam
cap = cv2.VideoCapture(0)

# Manual accuracy tracking (press 'y' = correct, 'n' = wrong)
total = 0
correct = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 350, 350
    roi = frame[y1:y2, x1:x2]

    # Preprocess image (convert BGR->RGB, resize, scale)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(roi_rgb, (128, 128)).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    preds = prediction[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Top-3 predictions for debugging
    top3_idx = np.argsort(preds)[-3:][::-1]
    top_labels = [f"{class_names[i]} ({preds[i]*100:.1f}%)" for i in top3_idx]
    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
    top_text = " | ".join(top_labels)

    # Draw rectangle & label
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Draw top-3 predictions and accuracy
    cv2.putText(frame, top_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    accuracy_text = f"Accuracy: {correct}/{total} ({(correct/total*100):.1f}%)" if total>0 else "Accuracy: N/A"
    cv2.putText(frame, accuracy_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('y'):
        total += 1
        correct += 1
    elif key == ord('n'):
        total += 1

cap.release()
cv2.destroyAllWindows()
