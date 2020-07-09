import glob
from skimage.feature import hog
from skimage import io
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import numpy as np
import cv2

data_label = {'across_edge':0,'edge':1,'fake':2,'region_edge':3}
class Dataset:
    def __init__(self,path):
        self.path = path
        self.train_set = {}
        self.test_set = {}
        self.classes = []
    def generate_sets(self):
        dataset_classes = glob.glob(self.path+'/*')
    
        for folder in dataset_classes:
            #print(folder)
            class_name = folder.split('\\')[-1]
            #print(class_name)
            rasterList = glob.glob(folder+'/*.jpg')
            i = data_label[class_name]
            self.train_set[i] = rasterList
            self.test_set[i] = rasterList[:10]

class get_HOG:
    def __init__(self):
        self.hog_list = []
        self.label = []
    def hog_descriptor(self,data):
        for label in range(len(data)): 
            for img_path in data[label]:
                img = cv2.imread(img_path)
                img = cv2.resize(img,(25,25))
                hot_img = hog(img,orientations=9,pixels_per_cell=(4, 4),
                              block_norm = 'L2',transform_sqrt = True,
                              cells_per_block=(2, 2),visualize=False, multichannel=True
                             )
                self.hog_list.append(hot_img)
                print('***************',hot_img.shape)
                self.label.append(label)
        
        
class Model:
        
    def train(self,feature,label):
        svm = LinearSVC()
        svm.fit(feature,label)
        joblib.dump(svm,'svm.dat')
        print('************finish!')
        
    def test(self,feature,label):
        svm = joblib.load('svm.dat')
        result = svm.predict(feature)
        print('&&&&&&&&&&&&&&',result)
        #print('&&&&&&&&&&&&&&&&&&&&&',label.shape)
        mask = result == label
        correct = np.count_nonzero(mask)
        print(mask.shape)
        accuracy = (correct *100.0 / result.size)
        print(accuracy)

if __name__ ==    '__main__':
    Train = False
    data = Dataset(r'F:\resnet_dataset_1\train')
    if len(data.train_set) == 0:
        data.generate_sets()
    hog_class = get_HOG()
    model = Model()
    #print(data.train_set)
    if Train:
        hog_class.hog_descriptor(data.train_set)
        train_feature = hog_class.hog_list
        
        train_label = hog_class.label
        model.train(train_feature,train_label )
    else:
        hog_class.hog_descriptor(data.test_set)
        test_feature = hog_class.hog_list
        #print(test_feature.shape)
        test_label = hog_class.label
        model.test(test_feature,test_label )
       