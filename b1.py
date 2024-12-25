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

            # Preprocess the frame
            img, ratio, (dw, dh) = letterbox(frame, new_shape=(imgsz, imgsz))  # Resize and pad to square
            img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange axes
            img = np.ascontiguousarray(img)

            img_tensor = torch.from_numpy(img).to(device).float()
            img_tensor /= 255.0  # Normalize to [0, 1]
            if half:
                img_tensor = img_tensor.half()
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = model(img_tensor)
            t2 = time_synchronized()

            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Generate segmentation masks
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

            # Resize masks back to the letterboxed frame size
            da_seg_mask_resized = cv2.resize(da_seg_mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
            ll_seg_mask_resized = cv2.resize(ll_seg_mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)

            # Undo letterbox padding
            height, width = frame.shape[:2]
            da_seg_mask_resized = da_seg_mask_resized[int(dh):int(dh + height), int(dw):int(dw + width)]
            ll_seg_mask_resized = ll_seg_mask_resized[int(dh):int(dh + height), int(dw):int(dw + width)]

            # Resize masks to match the original frame size
            da_seg_mask_resized = cv2.resize(da_seg_mask_resized, (width, height), interpolation=cv2.INTER_NEAREST)
            ll_seg_mask_resized = cv2.resize(ll_seg_mask_resized, (width, height), interpolation=cv2.INTER_NEAREST)

            # Overlay the masks on the frame
            color_seg = np.zeros_like(frame, dtype=np.uint8)
            color_seg[da_seg_mask_resized > 0] = [0, 255, 0]  # Green for driving area
            color_seg[ll_seg_mask_resized > 0] = [0, 0, 255]  # Red for lane lines
            frame = cv2.addWeighted(frame, 0.7, color_seg, 0.3, 0)

            # Process detections
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, frame, line_thickness=3)

            # Display the frame
            cv2.imshow('Webcam Detection', frame)
            # print("Original frame size:", frame.shape)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Currently, only webcam input is supported with this code version.")
        return


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
