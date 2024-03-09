import argparse
import boto3, cv2, time, numpy as np, matplotlib.pyplot as plt, random
from sagemaker.pytorch import PyTorchPredictor
from sagemaker.deserializers import JSONDeserializer

def main(endpoint_name):
    sm_client = boto3.client(service_name="sagemaker")
    
    endpoint_name = 'yolov8-pytorch-serverless-2024-03-08-16-35-37-360756'
    
    endpoint_created = False
    while True:
        response = sm_client.list_endpoints()
        for ep in response['Endpoints']:
            print(f"Endpoint Status = {ep['EndpointStatus']}")
            if ep['EndpointName']==endpoint_name and ep['EndpointStatus']=='InService':
                endpoint_created = True
                break
        if endpoint_created:
            break
        time.sleep(5)
   
    predictor = BoxerDetectionInference(endpoint_name=endpoint_name)

    filename = 'sm-notebook/Hagler_Mugabi_frame3000.jpg'
    orig_image = cv2.imread(filename)
 
    infer_start_time = time.time()
    
    result = predictor(orig_image)
    
    infer_end_time = time.time()
    
    print(f"Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")
    
    if 'boxes' in result:
        for idx,(x1,y1,x2,y2,conf,lbl) in enumerate(result['boxes']):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw Bounding Boxes
            color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
            cv2.rectangle(orig_image, (x1,y1), (x2,y2), color, 4)
            cv2.putText(orig_image, f"Class: {int(lbl)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(orig_image, f"Conf: {int(conf*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.show()

class BoxerDetectionInference:
    def __init__(self, endpoint_name):
        self.predictor = PyTorchPredictor(endpoint_name=endpoint_name,
                                 deserializer=JSONDeserializer())
       
    def __call__(self, frame):
        payload = self._bytes_from_file(frame)
        result = self.predict(payload)
        return result

    def _bytes_from_file(self, frame):
        payload = cv2.imencode('.jpg', frame)[1].tobytes()
        return payload

    def predict(self, payload):
        result = self.predictor.predict(payload)
        return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint_name')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(endpoint_name=args.endpoint_name)

