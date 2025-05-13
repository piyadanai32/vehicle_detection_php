import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import torch
import platform
import os
import requests
import time
import json
import logging

# กำหนด logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vehicle_detector")

def detect_gpu():
    """Detect and select the best available GPU"""
    gpu_info = {
        'device': 'cpu',
        'name': 'CPU',
        'backend': None
    }
    
    try:
        # Check NVIDIA GPU
        if torch.cuda.is_available():
            gpu_info['device'] = 'cuda'
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['backend'] = 'cuda'
            logger.info(f"NVIDIA GPU detected: {gpu_info['name']}")
            return gpu_info
            
        # Check AMD GPU on Windows
        elif platform.system() == 'Windows':
            try:
                import win32com.client
                wmi = win32com.client.GetObject("winmgmts:")
                gpu_list = wmi.InstancesOf("Win32_VideoController")
                for gpu in gpu_list:
                    if "AMD" in gpu.Name or "Radeon" in gpu.Name:
                        gpu_info['device'] = 'cpu'  # AMD uses CPU backend with ROCm
                        gpu_info['name'] = gpu.Name
                        gpu_info['backend'] = 'rocm'
                        logger.info(f"AMD GPU detected: {gpu_info['name']}")
                        return gpu_info
            except:
                pass
                
        # Check Intel GPU
        if torch.backends.mps.is_available():
            gpu_info['device'] = 'mps'
            gpu_info['name'] = 'Intel Graphics'
            gpu_info['backend'] = 'mps'
            logger.info(f"Intel GPU detected: {gpu_info['name']}")
            return gpu_info
            
    except Exception as e:
        logger.error(f"Error detecting GPU: {e}")
    
    logger.info("No dedicated GPU detected, using CPU")
    return gpu_info

# Detect available GPU
gpu_info = detect_gpu()

# Configure YOLO based on available GPU
try:
    # เพิ่มการตรวจสอบไฟล์โมเดล
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Downloading YOLOv8n model...")
        try:
            from ultralytics import download_from_hub
            download_from_hub("yolov8n.pt")
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            exit(1)
    
    model = YOLO(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

# Configure GPU settings
try:
    if gpu_info['backend'] == 'cuda':
        # NVIDIA GPU settings
        model.to('cuda')
        torch.backends.cudnn.benchmark = True
        logger.info("Using NVIDIA GPU acceleration")
    elif gpu_info['backend'] == 'rocm':
        # AMD GPU settings
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        logger.info("Using AMD GPU acceleration")
    elif gpu_info['backend'] == 'mps':
        # Intel GPU settings
        model.to('mps')
        model.model.half()
        logger.info("Using Intel GPU acceleration")
    else:
        # CPU settings
        model.to('cpu')
        torch.set_num_threads(4)
        logger.info("Using CPU acceleration")
except Exception as e:
    logger.error(f"Error configuring GPU: {e}")
    logger.info("Falling back to CPU")
    model.to('cpu')
    torch.set_num_threads(4)

# Enable OpenCV optimization
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Only track cars, motorcycles, and buses as requested
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
target_classes = ['car', 'motorcycle', 'bus']
tracker = Tracker()

# Video capture with resolution based on GPU capability
try:
    video_path = '2025-02-18 11-02-09.mkv'
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        exit(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        exit(1)
    logger.info("Video loaded successfully")
except Exception as e:
    logger.error(f"Error loading video: {e}")
    exit(1)

# Set resolution based on GPU capability
if gpu_info['backend'] in ['cuda', 'rocm']:
    # Higher resolution for dedicated GPUs
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
else:
    # Lower resolution for integrated GPU/CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize counters for each vehicle type
vehicle_counts = {
    'car': {'out': [], 'in': []},
    'motorcycle': {'out': [], 'in': []},
    'bus': {'out': [], 'in': []}
}

# Store vehicle types by ID
vehicle_types = {}  # {id: class_name}

# เพิ่ม dictionary สำหรับเก็บตำแหน่ง cx ก่อนหน้า
vehicle_last_cx = {}

# เพิ่ม dictionary สำหรับเก็บ state ของแต่ละ id ว่าอยู่ฝั่งไหนของเส้น
vehicle_states = {}

# เพิ่ม dictionary สำหรับเก็บจำนวนที่ส่งล่าสุด
last_sent_counts = {
    'car': {'out': 0, 'in': 0},
    'motorcycle': {'out': 0, 'in': 0},
    'bus': {'out': 0, 'in': 0}
}

# สำหรับเก็บเวลาสุดท้ายที่ส่งข้อมูล
last_send_time = time.time()


token = "dajsdkasjdsuad2348werwerewfjslfj8w424"
camera_id = 1  

# MODIFIED: Updated cropping function to get the bottom-right corner with custom dimensions
def get_custom_crop(frame):
    height, width = frame.shape[:2]
    
    # Calculate the starting point for the red area (half of width and height)
    red_start_x = width // 2
    red_start_y = height // 2
    
    # Calculate the dimensions of the orange area (smaller region within red area)
    # Adjust these percentages to match your orange box size
    orange_width_ratio = 0.7  # Percentage of red area width
    orange_height_ratio = 0.6  # Percentage of red area height
    
    # Calculate orange area dimensions
    orange_width = int((width - red_start_x) * orange_width_ratio)
    orange_height = int((height - red_start_y) * orange_height_ratio)
    
    # Calculate starting points for orange area (positioned in bottom right)
    orange_start_x = width - orange_width
    orange_start_y = height - orange_height
    
    # Return the cropped region (orange area)
    return frame[orange_start_y:height, orange_start_x:width]

# Create window
cv2.namedWindow("Vehicle Counter", cv2.WINDOW_NORMAL)

# FPS counter
frame_count = 0
start_time = cv2.getTickCount()

# GPU-specific batch size
batch_size = 4 if gpu_info['backend'] in ['cuda', 'rocm'] else 1

BACKEND_URL = "http://localhost:8000/vehicle_count/"

def send_count_to_backend(vehicle_type, direction, count):
    """
    ส่งข้อมูลการนับยานพาหนะไปยัง backend
    
    Args:
        vehicle_type (str): ประเภทยานพาหนะ (car, motorcycle, bus)
        direction (str): ทิศทาง (in, out)
        count (int): จำนวนที่นับได้
    """
    if count <= 0:
        return  # ไม่ส่งข้อมูลถ้าจำนวนเป็น 0 หรือน้อยกว่า
        
    try:
        payload = {
            "vehicle_type": vehicle_type,
            "direction": direction,
            "count": count,
            "token": token,      
            "camera_id": camera_id 
        }
        logger.info(f"Sending data to backend: {json.dumps(payload)}")
        
        response = requests.post(BACKEND_URL, json=payload, timeout=2)
        if response.status_code == 200:
            logger.info(f"Successfully sent data: {vehicle_type} {direction} count: {count}")
        else:
            logger.error(f"Failed to send data: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error - Unable to connect to {BACKEND_URL}")
    except Exception as e:
        logger.error(f"Error sending data to backend: {e}")

# เพิ่มฟังก์ชัน handle_exit เพื่อทำความสะอาดเมื่อปิดโปรแกรม
def handle_exit():
    """จัดการกับการออกจากโปรแกรม ทำความสะอาดและส่งข้อมูลที่เหลือ"""
    logger.info("Exiting program...")
    
    # ส่งค่าสุดท้ายที่ยังไม่ได้ส่ง
    for vehicle_type in target_classes:
        for direction in ['out', 'in']:
            current_count = len(set(vehicle_counts[vehicle_type][direction]))
            last_count = last_sent_counts[vehicle_type][direction]
            delta = current_count - last_count
            if delta > 0:
                send_count_to_backend(vehicle_type, direction, delta)
                logger.info(f"Final data sent: {vehicle_type} {direction} count: {delta}")
    
    # ปล่อยทรัพยากร
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Cleanup complete")

# ลูปหลักของโปรแกรม
try:
    while True:    
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video file reached")
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            logger.info(f"FPS: {fps:.2f}")
        
        # MODIFIED: Extract custom crop region (orange area)
        quadrant = get_custom_crop(frame)
        
        # Resize based on GPU capability
        if gpu_info['backend'] in ['cuda', 'rocm']:
            processed_frame = cv2.resize(quadrant, (600, 450), interpolation=cv2.INTER_AREA)
        else:
            processed_frame = cv2.resize(quadrant, (400, 300), interpolation=cv2.INTER_AREA)
        
        frame_height = processed_frame.shape[0]
        frame_width = processed_frame.shape[1]

        red_line_x = int(frame_width * 0.85)  
        blue_line_x = int(frame_width * 0.30)
        
        # คำนวณจุดเริ่มต้นและจุดสิ้นสุดสำหรับเส้น
        line_start_y_red = int(frame_height * 0.11)  
        line_end_y_red = int(frame_height * 0.54)    
        line_start_y_blue = int(frame_height * 0.40)  
        line_end_y_blue = int(frame_height * 40)
        
        try:
            # GPU-specific inference
            if gpu_info['backend'] == 'cuda':
                with torch.amp.autocast("cuda"):
                    results = model.predict(processed_frame, workers=4)
            else:
                results = model.predict(processed_frame, workers=2)
            
            result = results[0]
        except RuntimeError as e:
            logger.error(f"Inference error: {e}")
            continue
            
        if len(result.boxes) == 0:
            # Display the counts even when no vehicles are detected
            y_offset = 40
            cv2.putText(processed_frame, f'GPU: {gpu_info["name"]}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            for vehicle_type in target_classes:
                cv2.putText(processed_frame, f'{vehicle_type.capitalize()} Out: {len(set(vehicle_counts[vehicle_type]["out"]))}', 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                
                cv2.putText(processed_frame, f'{vehicle_type.capitalize()} In: {len(set(vehicle_counts[vehicle_type]["in"]))}', 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                
            # วาดเส้นที่มีความยาวน้อยลง แม้ไม่มีวัตถุถูกตรวจจับ
            cv2.line(processed_frame, (red_line_x, line_start_y_red), (red_line_x, line_end_y_red), (0, 0, 255), 2)
            cv2.line(processed_frame, (blue_line_x, line_start_y_blue), (blue_line_x, line_end_y_blue), (255, 0, 0), 2)
            
            cv2.imshow("Vehicle Counter", processed_frame)
            key = cv2.waitKey(1) & 0xFF 
            if key == 27:  # ESC key
                break
            continue
            
        # Convert detections to DataFrame
        a = result.boxes.data
        if gpu_info['backend'] == 'cuda':
            px = pd.DataFrame(a.cpu().numpy()).astype("float")
        else:
            px = pd.DataFrame(a.numpy()).astype("float")
        
        list = []
        for index, row in px.iterrows():
            class_id = int(row[5])
            if class_id < len(class_list):
                class_name = class_list[class_id]
                if class_name in target_classes:  # Only process target classes
                    box = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
                    list.append(box)
                    if len(list) > 0 and index < len(px):  # Save class name to use later
                        # We'll store the class name associated with this box when we get its ID
                        temp_class = class_name
        
        bbox_id = tracker.update(list)
        
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            
            # Find class for this ID
            for index, row in px.iterrows():
                box = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
                box_cx = (box[0] + box[2]) // 2
                box_cy = (box[1] + box[3]) // 2
                
                # If the center points are close, this is likely the same object
                if abs(cx - box_cx) < 10 and abs(cy - box_cy) < 10:
                    class_id = int(row[5])
                    if class_id < len(class_list):
                        vehicle_type = class_list[class_id]
                        if vehicle_type in target_classes:
                            vehicle_types[id] = vehicle_type
                            break
            
            # If we don't have a type for this ID yet, use the last known type
            if id not in vehicle_types:
                continue
            
            vehicle_type = vehicle_types[id]
            
            offset = 7

            # Draw vehicle type label on the bounding box
            cv2.rectangle(processed_frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
            cv2.putText(processed_frame, f'{vehicle_type}', (x3, y3-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- Begin: Robust counting logic with state ---
            # ตรวจสอบสถานะการข้ามเส้นแดง (ออก)
            prev_state_red = vehicle_states.get((id, 'red'), None)
            if cx < red_line_x:
                vehicle_states[(id, 'red')] = 'left'
            else:
                vehicle_states[(id, 'red')] = 'right'

            # ถ้าเคยอยู่ซ้ายแล้วข้ามไปขวา (ข้ามเส้นแดง)
            if prev_state_red == 'left' and vehicle_states[(id, 'red')] == 'right':
                if id not in vehicle_counts[vehicle_type]['out']:
                    vehicle_counts[vehicle_type]['out'].append(id)
                    cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)
                    logger.info(f"Vehicle ID {id} ({vehicle_type}) crossed RED line - OUT")

            # ตรวจสอบสถานะการข้ามเส้นน้ำเงิน (เข้า)
            prev_state_blue = vehicle_states.get((id, 'blue'), None)
            if cx > blue_line_x:
                vehicle_states[(id, 'blue')] = 'right'
            else:
                vehicle_states[(id, 'blue')] = 'left'

            # ถ้าเคยอยู่ขวาแล้วข้ามไปซ้าย (ข้ามเส้นน้ำเงิน)
            if prev_state_blue == 'right' and vehicle_states[(id, 'blue')] == 'left':
                if id not in vehicle_counts[vehicle_type]['in']:
                    vehicle_counts[vehicle_type]['in'].append(id)
                    cv2.circle(processed_frame, (cx, cy), 4, (255, 0, 0), -1)
                    logger.info(f"Vehicle ID {id} ({vehicle_type}) crossed BLUE line - IN")
            # --- End: Robust counting logic with state ---

        # --- ส่งข้อมูลทุกๆ 10 วินาที (เฉพาะจำนวนที่เพิ่มขึ้นในรอบนั้น) ---
        current_time = time.time()
        if current_time - last_send_time >= 10:
            logger.info("Sending 10-second interval data to backend...")
            for vehicle_type in target_classes:
                for direction in ['out', 'in']:
                    current_count = len(set(vehicle_counts[vehicle_type][direction]))
                    last_count = last_sent_counts[vehicle_type][direction]
                    delta = current_count - last_count
                    if delta > 0:
                        send_count_to_backend(vehicle_type, direction, delta)
                    last_sent_counts[vehicle_type][direction] = current_count
            last_send_time = current_time

        # วาดเส้นที่มีความยาวน้อยลง
        cv2.line(processed_frame, (red_line_x, line_start_y_red), (red_line_x, line_end_y_red), (0, 0, 255), 2)
        cv2.line(processed_frame, (blue_line_x, line_start_y_blue), (blue_line_x, line_end_y_blue), (255, 0, 0), 2)
        
        # Add counts to display
        y_offset = 40
        cv2.putText(processed_frame, f'GPU: {gpu_info["name"]}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        for vehicle_type in target_classes:
            cv2.putText(processed_frame, f'{vehicle_type.capitalize()} Out: {len(set(vehicle_counts[vehicle_type]["out"]))}', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(processed_frame, f'{vehicle_type.capitalize()} In: {len(set(vehicle_counts[vehicle_type]["in"]))}', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        cv2.imshow("Vehicle Counter", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    # จัดการการปิดโปรแกรมอย่างสวยงาม
    handle_exit()

except KeyboardInterrupt:
    logger.info("Program interrupted by user")
    handle_exit()
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    handle_exit()