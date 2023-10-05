import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv
import mysql.connector

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="shop"
)


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.bottle_added = False
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
       
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
         # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls[0]
            conf = boxes.conf[0]
            xyxy = boxes.xyxy[0]

            if class_id == 0.0:
          
              xyxys.append(result.boxes.xyxy.cpu().numpy())
              confidences.append(result.boxes.conf.cpu().numpy())
              class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} "
              for confidence, class_id in zip(detections.confidence, detections.class_id)]

        print("product found" , self.labels)
        if "bottle " in self.labels:
           # print("Good job")
            if not self.bottle_added:

                frame_height = frame.shape[0]
                green_line_y = frame_height - 200
                cv2.rectangle(frame, (0, green_line_y), (frame.shape[1], frame_height), (0, 255, 0), 3)
                for xyxy, label in zip(detections.xyxy, self.labels):
                    _, _, _, y2 = xyxy.tolist()
                    if label.strip() == "bottle" and y2 > green_line_y:
                        print("Bottle added to cart")
                        
                        mycursor = mydb.cursor()
                        
                        sql = "INSERT INTO cart(item_id,item_name,item_price) VALUES (%s, %s,%s)"
                        val = (1,"bottle",40)
                        mycursor.execute(sql, val)
                        mydb.commit()

                        print(mycursor.rowcount, "record inserted.")
                        
                        self.bottle_added = True
                                    

        
       # if "book " in self.labels:
           # print("Good job")
    
        #    frame_height = frame.shape[0]
         #   green_line_y = frame_height - 100
         #   cv2.rectangle(frame, (0, green_line_y), (frame.shape[1], frame_height), (0, 255, 0), 3)
         #   for xyxy, label in zip(detections.xyxy, self.labels):
          #      _, _, _, y2 = xyxy.tolist()
          #     if label.strip() == "book" and y2 > green_line_y:
          #          print("book added to cart")
                    
            
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=0)
detector()
