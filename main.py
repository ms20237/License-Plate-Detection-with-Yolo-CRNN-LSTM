import os
import cv2
import torch
import argparse

from ultralytics import YOLO
from CRNN import CRNN
from utils import preprocess_plate, decode_predictions


def init():
    parser = argparse.ArgumentParser(description="Analyze Sqrt Area FiftyOne dataset.")
    parser.add_argument('--detect_model_path',
                        type=str,
                        required=True,
                        help="Path of plate detect model(yolo, ...).")
    
    parser.add_argument('--recg_model_path',
                        type=str,
                        required=True,
                        help="Path of Recognition model(LSTM).")
    
    parser.add_argument('--video_path',
                        type=str,
                        required=True,
                        help="Path of test Video.")
    
    parser.add_argument('--output_path',
                        type=str,
                        default="./output",
                        help="Path of output Video.")
    
    return parser.parse_args()


def run(detect_model_path: str,
        recg_model_path: str,
        video_path: str,
        output_path: str = None):
    
    ## Load the trained models 
    # Detect Plate model
    print("Loading models...")
    try:
        yolo_model = YOLO(detect_model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print(f"And the model file '{detect_model_path}' is available.")
        return
    
    # Add lstm model for recognizing letters    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_classes = len(alphabet) + 1  # CTC blank

    crnn = CRNN(num_classes).to(device)
    if recg_model_path:
        crnn.load_state_dict(torch.load(recg_model_path, map_location=device))
        crnn.eval()
    else:
        print("Warning: No LSTM/CRNN model provided.")

    
    ## Initialize video capture and data structures
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'. It may be corrupted or in an unsupported format.")
        return
    
    ## Setup Video Writer for saving the output
    writer = None
    if output_path:
        if not output_path.endswith((".mp4", ".avi")):
            output_path = os.path.join(output_path, "result.mp4")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output_path.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_path.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            print("Warning: Unsupported output format. Defaulting to .mp4")
            output_path += '.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output video will be saved to: {output_path}")
        
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Original resolution: {orig_width}x{orig_height}")    

    ## Start Capturing frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detect_frames = yolo_model(frame, verbose=False)
        
        results = detect_frames[0]
        # print("Detected boxes:", len(results.boxes))
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            # print("Crop shape:", plate_crop.shape)

            if plate_crop.size == 0:
                continue

            plate_tensor = preprocess_plate(plate_crop).to(device)
            # print("Plate tensor shape:", plate_tensor.shape)

            with torch.no_grad():
                preds = crnn(plate_tensor)
                pred_indices = preds.argmax(2)
                
                # plate_text = ctc_decode(preds, alphabet)
                plate_text = decode_predictions(preds, alphabet)[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, plate_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        annotated_frame = frame
        if writer:
            writer.write(annotated_frame)
        
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        cv2.imshow('YOLO + LSTM Recognition', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    ## Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Inference complete.")
    
    
if __name__ == "__main__":
    args = init()
    run(args.detect_model_path,
        args.recg_model_path,
        args.video_path,
        args.output_path,
    )   
    
    