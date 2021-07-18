import cv2 
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fastseg import MobileV3Small
from fastseg.image import colorize, blend




def yolo(img):
    '''
    Parameters:
        img:Input image to detect objects from
    Function:
        Classifies the objects in an Image
    Returns:
        models.Detection class object
    
    '''
    # image = Image.open(img)
    output = model(img)
    return output

def segmentation(img):
    '''
    Parameters:
        img:Input image to segment
    Function:
        Segments the image
    Returns:
        PIL.Image
    '''
    img_resize=img.resize((852, 480))
    output=model_seg.predict_one(img_resize)
    output_img=colorize(output)
    blend_image=blend(img_resize,output_img)
    
    return blend_image  

def plot_one_box_PIL(box, im, color=(128, 128, 128), label=None, line_thickness=None):
    '''
    Parameters: 
        box: list, xmin, ymin, xmax, ymax: list
        im: PIL format, Image in the PIL format
        color: tupple, RGB format of color of the box
        label: string, label of the box
        line thickness: int, thickness of the box

    Function: 
        Draws one box on the PIL image im

    Returns:
        numpy array
    '''
    
    # Plots one bounding box on image 'im' using PIL
    
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness or max(int(min(im.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=color)  # plot
    if label:
        font = ImageFont.truetype("arial.ttf", size=max(round(max(im.size) / 70), 12))
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)

if __name__ == '__main__':


    #Dict of indices and the respective label
    custom_coco_dictt = {0: 'person',
                        1: 'bicycle',
                        2: 'car',
                        3: 'motorcycle',
                        4: 'bus',
                        5: 'truck',
                        6: 'fire hydrant',
                        7: 'stop sign',
                        8: 'bench',
                        9: 'bird',
                        10: 'cat',
                        11: 'dog',
                        12: 'horse',
                        13: 'sheep',
                        14: 'cow',
                        15: 'chair',
                        16: 'tree',
                        17: 'pothole',
                        18: 'obstacle'}
    #Color code for respective class
    color_codes = [
                        (0, 255, 255), 
                        (128, 128, 128), 
                        (0, 0, 128), 
                        (192, 192, 192), 
                        (0, 0, 0), 
                        (0, 128, 0), 
                        (128, 128, 0), 
                        (0, 128, 128), 
                        (0, 0, 255), 
                        (0, 255, 0), 
                        (128, 0, 128), 
                        (255, 255, 255), 
                        (128, 0, 0), 
                        (255, 0, 0), 
                        (255, 255, 0), 
                        (255, 0, 255), 
                        (0, 100, 0), 
                        (255, 165, 0), 
                        (255, 239, 213) 
                    ]


    #Load the yolov5s model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False,pretrained=False, classes=19)
    model.load_state_dict(torch.load('best.pt')['model'].state_dict()) #weights for object detection
    model = model.fuse().autoshape()

    #Load the MovileNetV3 model
    model_seg = MobileV3Small(5)
    model_seg.load_state_dict(torch.load("model_small_latest.pt")) #weights for image segmentation
    model_seg =model_seg.cuda().eval()

    #Capture the video
    cap = cv2.VideoCapture("/home/shreyas/yolov5/2019_0612_024916_005.MOV")
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
   
    size = (frame_width, frame_height)

    result = cv2.VideoWriter('mobile_net_latest.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    
    while True:
        ret, img = cap.read()

        if cv2.waitKey(1) == 27:
            break
        
        #Convert image to PIL format
        img = Image.fromarray(img)


        output_segmentation=segmentation(img) 
        output_yolo=yolo(img) 
        output_seg=output_segmentation.resize((1920,1080),Image.NEAREST)

        #get the box coordinates and convert it into dictionary        
        box_cords = output_yolo.pandas().xyxy[0].to_dict('index')
        
    
        for key in box_cords.keys():
            *box, conf, cls, name = box_cords[key].values()
            img_asarray = plot_one_box_PIL(box, output_seg, color = color_codes[int(cls)], label = custom_coco_dictt[int(name)]+ " " + str(round(conf, 2)), line_thickness = 2)
           
        result.write(img_asarray)
        cv2.imshow("OUT", img_asarray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    result.release()
    cv2.destroyAllWindows()
