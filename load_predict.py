import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

MODEL_PATH = "best_model.pth"

# Function to load model
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path,
                       map_location=device)
    return model

# Function to preprocess image
def preprocess(img):
    img = Image.fromarray(np.uint8(img))
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    preprocess_img = preprocess(img)
    batch_img_tensor = torch.unsqueeze(preprocess_img, 0)
    return batch_img_tensor

# Function to make predictions
def predict(img, path=MODEL_PATH):
    batch_img_tensor = preprocess(img)
    model = load_model(path)
    labels = ['mask', 'no mask']
    model.eval()
    out = model(batch_img_tensor)
    prob = torch.nn.functional.softmax(out, dim=1)[0]
    index = torch.argmax(torch.nn.functional.softmax(out, dim=-1))
    label = labels[index]
    probability = round(prob[index].item(), 2)
    if label == 'mask':
        result = 1
    else:
        result = 0
    return result, probability

# Live Demo capturing facemask using OpenCV
results={0:'no mask', 1:'mask'}
GR_dict={0:(0,0,255), 1:(0,255,0)}
rect_size = 4
cap = cv2.VideoCapture(0) 

haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1) 
    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        label, prob = predict(face_img)
        text = f"{results[label]}: {prob}"
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break
cap.release()
cv2.destroyAllWindows()