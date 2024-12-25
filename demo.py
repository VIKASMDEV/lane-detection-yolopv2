import argparse
import time
import cv2
import torch
import numpy as np

# Import required utilities
from utils.utils import (
    time_synchronized, select_device, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, letterbox, scale_coords,non_max_suppression
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source (0 for webcam, file/folder for others)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    return parser


def detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    device = select_device(opt.device)
    half = device.type != 'cpu'  # Use FP16 precision if supported

    # Load the YOLOP model
    model = torch.jit.load(weights)
    model.to(device)
    if half:
        model.half()
    model.eval()

    # Open webcam or video source
    if source.isdigit() or source == '0':  # Webcam source
        print("Using webcam...")
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError("Error: Unable to open the webcam.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from the webcam.")
                break

            # Resize the frame to match mask dimensions (1280x720)
            frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

            # Preprocess the resized frame
            img, ratio, (dw, dh) = letterbox(frame_resized, new_shape=(imgsz, imgsz))  # Resize and pad to square
            img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange axes
            img = np.ascontiguousarray(img)

            img_tensor = torch.from_numpy(img).to(device).float()
            img_tensor /= 255.0  # Normalize to [0, 1]
            if half:
                img_tensor = img_tensor.half()
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Inference
            [pred, anchor_grid], seg, ll = model(img_tensor)

            # Generate segmentation masks
            da_seg_mask = driving_area_mask(seg)  # Drivable area mask
            ll_seg_mask = lane_line_mask(ll)  # Lane line mask

            # Compute control commands
            control_commands = compute_control_commands(da_seg_mask, ll_seg_mask)
            print("Control Commands:", control_commands)

            # Display the frame (optional, for debugging)
            color_seg = np.zeros_like(frame_resized, dtype=np.uint8)
            color_seg[da_seg_mask > 0] = [0, 255, 0]  # Green for driving area
            color_seg[ll_seg_mask > 0] = [0, 0, 255]  # Red for lane lines
            frame_with_masks = cv2.addWeighted(frame_resized, 0.7, color_seg, 0.3, 0)
            cv2.imshow('Webcam Detection', frame_with_masks)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Currently, only webcam input is supported with this code version.")
        return


def compute_control_commands(da_seg_mask, ll_seg_mask):
    """
    Compute the control commands based on the drivable area and lane line masks.

    :param da_seg_mask: Drivable area mask.
    :param ll_seg_mask: Lane line mask.
    :return: Dictionary with control commands (e.g., steer, speed).
    """
    h, w = da_seg_mask.shape

    # Find the center of the drivable area
    da_center = np.mean(np.where(da_seg_mask > 0), axis=1) if np.any(da_seg_mask > 0) else (h // 2, w // 2)

    # Estimate lane position (assuming two lanes)
    lane_indices = np.where(ll_seg_mask > 0)
    if len(lane_indices[1]) > 0:
        lane_center = np.mean(lane_indices[1])
    else:
        lane_center = w // 2  # Default to center if no lane detected

    # Compute steering command (negative for left, positive for right)
    steer = (lane_center - w // 2) / (w // 2)  # Normalize to [-1, 1]

    # Speed is reduced if not centered in the drivable area
    distance_to_center = abs(da_center[1] - w // 2) / (w // 2)
    speed = max(0.1, 1.0 - distance_to_center)  # Normalize to [0.1, 1.0]

    return {
        "steer": steer,
        "speed": speed
    }


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
