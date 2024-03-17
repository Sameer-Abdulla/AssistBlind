import cv2
import torch
import matplotlib.pyplot as plt 
import io
import numpy as np





# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

def plt_to_jpeg():
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf.getvalue()

# Hook into OpenCV
def depthFunc(cap):
    while True: 
        ret, frame = cap.read()

        # Transform input for midas 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to('cpu')

        # Make a prediction
        with torch.no_grad(): 
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = img.shape[:2], 
                mode='bicubic', 
                align_corners=False
            ).squeeze()

            output = prediction.cpu().numpy()
            output = (output - output.min()) / (output.max() - output.min()) * 255
            output = output.astype(np.uint8)
            #print(output)
        plt.imshow(output, cmap='jet')  # cmap='jet' for better visualization
        plt.axis('off')  # Turn off axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust margins
        plt.gca().set_axis_off()  # Turn off axis
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Hide x axis
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Hide y axis

        depth_jpeg = plt_to_jpeg()
        frame = cv2.imdecode(np.frombuffer(depth_jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
        
        return depth_jpeg
    
    