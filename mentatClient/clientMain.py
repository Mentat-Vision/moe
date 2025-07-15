import cv2
import asyncio
import websockets
import websockets.exceptions
import json
import base64
import numpy as np
from datetime import datetime
import time
import threading
import os

def load_config_and_cameras():
    config = {"ENABLE_WINDOW_PREVIEW": True, "SERVER_IP": "10.8.162.58", "SERVER_PORT": "5000"}
    cameras = {}
    if os.path.exists("config.env"):
        with open("config.env") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                key, value = line.split("=", 1)
                value = value.split("#")[0].strip()
                if key == "ENABLE_WINDOW_PREVIEW":
                    config[key] = value.lower() == "true"
                elif key in ["SERVER_IP", "SERVER_PORT"]:
                    config[key] = value
                elif key.startswith("CAMERA_"):
                    name = key[7:]
                    cameras[name] = int(value) if value.isdigit() else value
    if not cameras:
        cameras = {"webcam_0": 0, "webcam_1": 1}
    return config, cameras

class MultiCameraClient:
    def __init__(self):
        self.config, self.cameras = load_config_and_cameras()
        if not self.cameras: raise ValueError("No cameras enabled. Check config.env file.")
        self.websockets, self.connected = {}, {}
        self.yolo_data, self.blip_data = {}, {}
        self.colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,128),(255,165,0)]
        self.last_yolo_time, self.last_blip_time = {}, {}
        self.yolo_interval, self.blip_interval = 0.2, 3.0
        self.camera_status = {}
        for cam in self.cameras:
            self.yolo_data[cam] = {"detections": [], "person_detections": [], "person_count": 0, "fps": 0}
            self.blip_data[cam] = {"caption": "", "fps": 0}
            self.connected[cam] = False
            self.last_yolo_time[cam] = self.last_blip_time[cam] = 0
            self.camera_status[cam] = {"working": True, "failures": 0}
        print(f"🖥️ Window preview: {'ENABLED' if self.config['ENABLE_WINDOW_PREVIEW'] else 'DISABLED'}")

    async def connect_to_server(self, camera_name):
        try:
            url = f"ws://{self.config['SERVER_IP']}:{self.config['SERVER_PORT']}"
            self.websockets[camera_name] = await websockets.connect(url)
            self.connected[camera_name] = True
            print(f"🔌 Camera {camera_name} connected to server: {url}")
            return True
        except Exception as e:
            print(f"❌ Camera {camera_name} failed to connect: {e}")
            return False

    def open_camera(self, camera_name, camera_source):
        cap = cv2.VideoCapture(camera_source)
        if isinstance(camera_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 25)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_name} ({camera_source})")
            return None
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera {camera_name} opened ({w}x{h})")
        return cap

    async def send_frame_to_expert(self, camera_name, frame, expert_type):
        if not self.connected[camera_name] or camera_name not in self.websockets:
            return
        try:
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            message = {"expert": expert_type, "camera_id": camera_name, "frame": frame_base64}
            await self.websockets[camera_name].send(json.dumps(message))
            timeout = 5.0 if expert_type == "BLIP" else 2.0
            response = await asyncio.wait_for(self.websockets[camera_name].recv(), timeout=timeout)
            results = json.loads(response)
            if expert_type == "YOLO" and "error" not in results:
                yd = self.yolo_data[camera_name]
                yd.update({k: results.get(k, yd[k]) for k in ["detections", "person_detections", "person_count", "fps"]})
                if yd["detections"]:
                    labels = [f"{d['class']} ({d['confidence']:.2f})" for d in yd["detections"]]
                    print(f"🎯 {camera_name} - {datetime.now().strftime('%H:%M:%S')} - {', '.join(labels)} (FPS: {yd['fps']}, Persons: {yd['person_count']})")
            elif expert_type == "BLIP" and "error" not in results:
                bd = self.blip_data[camera_name]
                bd.update({k: results.get(k, bd[k]) for k in ["caption", "fps"]})
                if bd["caption"]:
                    print(f"📝 {camera_name} - {datetime.now().strftime('%H:%M:%S')} - {bd['caption']} (FPS: {bd['fps']})")
            elif "error" in results:
                print(f"❌ {camera_name} {expert_type} error: {results['error']}")
        except asyncio.TimeoutError:
            print(f"⏰ {camera_name} {expert_type} timeout")
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 {camera_name} connection closed, reconnecting...")
            self.connected[camera_name] = False
            await self.connect_to_server(camera_name)
        except Exception as e:
            print(f"❌ {camera_name} {expert_type} error: {e}")

    def draw_yolo_detections(self, frame, camera_name):
        detections = self.yolo_data[camera_name]["detections"]
        h, w = frame.shape[:2]
        sx, sy = w / 640.0, h / 480.0
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = [int(d["bbox"][j] * (sx if j%2==0 else sy)) for j in range(4)]
            color = self.colors[i % len(self.colors)]
            label = f"{d['class']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def draw_person_ids(self, frame, camera_name):
        persons = self.yolo_data[camera_name]["person_detections"]
        h, w = frame.shape[:2]
        sx, sy = w / 640.0, h / 480.0
        for p in persons:
            if "id" in p:
                x1, y1, x2, y2 = [int(p["bbox"][j] * (sx if j%2==0 else sy)) for j in range(4)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"ID: {p['id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    def draw_blip_caption(self, frame, camera_name):
        caption = self.blip_data[camera_name]["caption"]
        if caption:
            words, lines, cur = caption.split(), [], ""
            for word in words:
                if len(cur + " " + word) < 40:
                    cur += (" " + word) if cur else word
                else:
                    lines.append(cur)
                    cur = word
            if cur: lines.append(cur)
            h = frame.shape[0]
            y = h - 80
            for i, line in enumerate(lines[:3]):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (10, y-th-5), (10+tw+10, y+5), (0,0,0), -1)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                y += 25

    def draw_status_info(self, frame, camera_name):
        w = frame.shape[1]
        x = w - 200
        cv2.putText(frame, f"Camera: {camera_name}", (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        status = "Connected" if self.connected[camera_name] else "Disconnected"
        color = (0,255,0) if self.connected[camera_name] else (0,0,255)
        cv2.putText(frame, f"Server: {status}", (x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y = 80
        if self.yolo_data[camera_name]["fps"] > 0:
            cv2.putText(frame, f"YOLO FPS: {self.yolo_data[camera_name]['fps']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25
        if self.blip_data[camera_name]["fps"] > 0:
            cv2.putText(frame, f"BLIP FPS: {self.blip_data[camera_name]['fps']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25
        if self.yolo_data[camera_name]["person_count"] > 0:
            cv2.putText(frame, f"Persons: {self.yolo_data[camera_name]['person_count']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    async def run_async(self):
        for cam in self.cameras:
            await self.connect_to_server(cam)
        caps = {cam: self.open_camera(cam, src) for cam, src in self.cameras.items() if self.open_camera(cam, src)}
        if not caps:
            print("❌ No cameras could be opened. Check your configuration.")
            return
        print("🎥 Multi-Camera Client running with central server architecture.")
        print("Press 'q' to quit." if self.config["ENABLE_WINDOW_PREVIEW"] else "Running in headless mode. Ctrl+C to quit.")
        while True:
            now = time.time()
            for cam in list(caps):
                if not self.camera_status[cam]["working"]: continue
                cap = caps[cam]
                ret, frame = cap.read()
                if not ret:
                    self.camera_status[cam]["failures"] += 1
                    if self.camera_status[cam]["failures"] > 10:
                        print(f"❌ Camera {cam} failed too many times, disabling")
                        self.camera_status[cam]["working"] = False
                        cap.release()
                        del caps[cam]
                    continue
                self.camera_status[cam]["failures"] = 0
                if now - self.last_yolo_time[cam] >= self.yolo_interval:
                    await self.send_frame_to_expert(cam, frame, "YOLO")
                    self.last_yolo_time[cam] = now
                if now - self.last_blip_time[cam] >= self.blip_interval:
                    await self.send_frame_to_expert(cam, frame, "BLIP")
                    self.last_blip_time[cam] = now
                if self.config["ENABLE_WINDOW_PREVIEW"]:
                    self.draw_yolo_detections(frame, cam)
                    self.draw_person_ids(frame, cam)
                    self.draw_blip_caption(frame, cam)
                    self.draw_status_info(frame, cam)
                    display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                    cv2.imshow(f"Camera {cam}", display_frame)
            if self.config["ENABLE_WINDOW_PREVIEW"]:
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                await asyncio.sleep(0.01)
        for cap in caps.values(): cap.release()
        if self.config["ENABLE_WINDOW_PREVIEW"]: cv2.destroyAllWindows()
        for ws in self.websockets.values(): await ws.close()

def main():
    try:
        client = MultiCameraClient()
        asyncio.run(client.run_async())
    except ValueError as e:
        print(f"❌ {e}\n💡 To enable cameras, edit config.env and uncomment the cameras you want to use.")
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user (Ctrl+C)")

if __name__ == "__main__":
    main()