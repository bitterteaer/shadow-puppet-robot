import sys, os.path
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
import yaml
from yaml.loader import SafeLoader
import colorsys
import random

class Detector:
    def __init__(self,pdmodel_path, pdmodel_file, pdmodel_config, device, img_size = 640) -> None:
        
        self.pdmodel_path = pdmodel_path
        self.pdmodel_file = os.path.join(pdmodel_path,pdmodel_file)
        self.pdmodel_config = os.path.join(pdmodel_path,pdmodel_config)
        print(self.pdmodel_file,self.pdmodel_config)
        self.device = device
        self.labellist = []
        #加载标签列表
        label_list=[]
        self.img_size = img_size
        with open(self.pdmodel_config) as f:
            data = yaml.load(f, Loader=SafeLoader)
        self.label_list = data['label_list'];
        self.colors = self.ncolors(len(self.label_list))
        #读取模型结构
        self.ie = IECore()
        self.net = self.ie.read_network(self.pdmodel_file)
        #调整输入层尺寸
        self.net.reshape({'image': [1, 3, self.img_size, self.img_size], 'im_shape': [
                1, 2], 'scale_factor': [1, 2]})
        #加载预训练权重
        self.exec_net = self.ie.load_network(self.net, self.device)
        assert isinstance(self.exec_net, ExecutableNetwork)
    
    #检测并绘制结果
    def detect(self,image,conf_thresh=0.2, showImage = True, frameName = None, saveImage = False, saveImgName = None):
        result_image = image.copy()
        test_image = self.image_preprocess(image, self.img_size)
        test_im_shape = np.array([[self.img_size, self.img_size]]).astype('float32')
        test_scale_factor = np.array([[1, 2]]).astype('float32')
        #构建输入字典
        inputs_dict = {'image': test_image, "im_shape": test_im_shape,
                    "scale_factor": test_scale_factor}
        #模型推理
        output = self.exec_net.infer(inputs_dict)
        result_ie = list(output.values())
        #绘制并保存结果    
        scale_x = result_image.shape[1]/self.img_size*2
        scale_y = result_image.shape[0]/self.img_size
        result_image = self.draw_box(result_image, result_ie[0], scale_x, scale_y,conf_thresh = conf_thresh)
        if(showImage and frameName != None):
            cv2.imshow(frameName,result_image)
        elif(showImage and frameName == None):
            cv2.imshow("openvino",result_image)
        if (saveImage and saveImgName != None):
            cv2.imwrite(saveImgName,result_image)
        elif(saveImage and saveImgName == None):
            cv2.imwrite("openvino_test.jpg",result_image)
 
    #图像预处理，主要负责调整输入图像尺寸，并归一化
    def image_preprocess(self, input_image, size):
        img = cv2.resize(input_image, (size,size))
        img = np.transpose(img, [2,0,1]) / 255
        img = np.expand_dims(img, 0)
        ##NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
        img_mean = np.array([0.485, 0.456,0.406]).reshape((3,1,1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
        img -= img_mean
        img /= img_std
        return img.astype(np.float32)
    #生成具有辨识度的类别颜色
    def get_n_hls_colors(self,num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    def ncolors(self,num):
        rgb_colors = []
        if num < 1:
            return rgb_colors
        hls_colors = self.get_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
            rgb_colors.append([r, g, b])
        return rgb_colors
    #将检测结果绘制在图像中
    def draw_box(self,img, results, scale_x, scale_y, line_width=2 , conf_thresh = 0.2):
        for i in range(len(results)):
            # print(results[i])
            bbox = results[i, 2:]
            label_id = int(results[i, 0])
            score = results[i, 1]
            if(score>conf_thresh):
                xmin, ymin, xmax, ymax = [int(bbox[0]*scale_x), int(bbox[1]*scale_y), 
                                        int(bbox[2]*scale_x), int(bbox[3]*scale_y)]
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),self.colors[label_id],3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_text = self.label_list[label_id];
                tf = max(line_width - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label_text, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
                cv2.rectangle(img, (xmin, ymin), (xmin+w, ymin-h-3), self.colors[label_id], -1)
                cv2.putText(img, label_text,(xmin,ymin), font, line_width / 3,(0,0,0), thickness=tf, lineType=cv2.LINE_AA)
                # cv2.putText(img, str(score),(xmin,ymin-40), font, 0.8,(255,255,255), 2,cv2.LINE_AA)
        return img

def main():
    # 指定检测模型所在路径
    pdmodel_path = "picodet_v2_s_192_pedestrian"
    pdmodel_file = "model.pdmodel"
    pdmodel_config = "infer_cfg.yml"
    device = 'CPU' 
    img_size = 640 #图像尺寸
    detNet = Detector(pdmodel_path,pdmodel_file,pdmodel_config,device,img_size)
    #读取图像并进行预处理
    image_path = "img_test.png"
    input_image = cv2.imread(image_path)
    detNet.detect(input_image,saveImage=True)
 
if __name__ == '__main__':
    sys.exit(main())