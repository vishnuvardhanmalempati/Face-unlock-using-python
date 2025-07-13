import argparse, pathlib, sys, cv2, numpy as np

DATA_DIR = pathlib.Path("face_db")
MODEL_FILE = pathlib.Path("face_model.yml")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

LEN_FACESET = 100  # ⬅️ Now capturing 100 images
CONF_THRESHOLD = 55

def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "user").mkdir(exist_ok=True)

def capture_faces(label: str) -> None:
    cap = cv2.VideoCapture(0)
    count = 0
    print("[INFO] Look straight at the webcam. Capturing 100 face images…")
    while count < LEN_FACESET and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            cv2.putText(frame, "Face not detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            path = DATA_DIR / label / f"{count:03d}.png"
            cv2.imwrite(str(path), face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{LEN_FACESET}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Enrolment – press q to quit early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()
    print(f"[DONE] Captured {count} images.")

def train_model() -> None:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []
    for img_path in sorted((DATA_DIR / "user").glob("*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(0)
    recognizer.train(images, np.array(labels))
    recognizer.save(str(MODEL_FILE))
    print(f"[MODEL] Trained and saved to {MODEL_FILE}")

def unlock() -> None:
    if not MODEL_FILE.exists():
        sys.exit("[ERROR] No trained model. Run --enrol first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    cap = cv2.VideoCapture(0)
    unlocked = False
    print("[INFO] Face Unlock is running. Press q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            cv2.putText(frame, "Face not detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, conf = recognizer.predict(face)

            if label == 0 and conf < CONF_THRESHOLD:
                cv2.putText(frame, "Face matching ", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "Succes!", (x, y+h+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                unlocked = True
            else:
                cv2.putText(frame, "Face not matching! Try Again", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Face Unlock Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser(description="Simple Face‑Unlock project")
    p.add_argument("--enrol", action="store_true", help="Capture faces and train model")
    p.add_argument("--unlock", action="store_true", help="Run live face unlock")
    args = p.parse_args()

    ensure_dirs()
    if args.enrol:
        capture_faces("user")
        train_model()
    elif args.unlock:
        unlock()
    else:
        p.print_help()

if __name__ == "__main__":
    main()
