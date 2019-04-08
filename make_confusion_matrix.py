from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
#import ImageDraw
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def get_re_list(img_re_path,img_pr_path):
    img1 = np.array(Image.open(img_pr_path))
    w,h = img1.shape
    list_pr = list(img1.reshape(1, w*h)[0])
    print(len(list_pr))
    img2 = np.array(Image.open(img_re_path))
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    list_re = list(img2.reshape(1, w*h)[0])
    print(len(list_re))
    return list_re, list_pr

def get_all_list(re_dir, pr_dir):
    y_true = np.array([[]])
    y_pred = np.array([[]])
    for file in os.listdir(re_dir):
        img1 = np.array(Image.open(re_dir + '/' + file))
        img2 = np.array(Image.open(pr_dir + '/' + file))
        w,h = img2.shape
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        #print(img1.shape)
        y_true= np.append(y_true, img1)
        y_pred= np.append(y_pred, img2)
        #print(y_true.size)
        #print(y_pred.size)
    list_re = list(y_true)
    list_pr = list(y_pred)
    return list_re, list_pr
    
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary): 
        plt.imshow(cm, interpolation='nearest', cmap=cmap) 
        plt.title(title) 
        plt.colorbar() 
        xlocations = np.array(range(len(labels))) 
        plt.xticks(xlocations, labels, rotation=90) 
        plt.yticks(xlocations, labels) 
        plt.ylabel('True label') 
        plt.xlabel('Predicted label') 

#'/home/hujun/data/pred-mask/23类/VGG3-PSP-1600/00000000000000680-1.png'
def make_confusion_matrix(re_dir, pr_dir, save_dir):
    #labels表示你不同类别的代号，比如这里的demo中有23个类别 
    count = 0
    #y_true代表真实的label值 y_pred代表预测得到的lavel值 
    for file in os.listdir(re_dir):
        #print(file)
        y_true, y_pred= get_re_list(re_dir + '/' + file, pr_dir + '/' + file)
        tick_marks = np.array(range(len(labels))) + 0.5
        cm = confusion_matrix(y_true, y_pred) 
        np.set_printoptions(precision=2) 
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        #print(cm_normalized)
        plt.figure(figsize=(12, 8), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array) 
        try:
            for x_val, y_val in zip(x.flatten(), y.flatten()): 
                #print(x_val, y_val)
                c = cm_normalized[y_val][x_val] 
                if c > 0.01: 
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7.5, va='center', ha='center')
                    # offset the tick 
            plt.gca().set_xticks(tick_marks, minor=True) 
            plt.gca().set_yticks(tick_marks, minor=True) 
            plt.gca().xaxis.set_ticks_position('none') 
            plt.gca().yaxis.set_ticks_position('none') 
            plt.grid(True, which='minor', linestyle='-') 
            plt.gcf().subplots_adjust(bottom=0.15) 
            plt.plot([0,22], [0,22], color="red", linestyle='--', linewidth = 0.5)
            #plt.plot([0,21],[0,21])
            #其中average参数有五种：(None, ‘micro’, ‘macro’, ‘weighted’, ‘samples’)
            #宏平均（Macro-averaging），是先对每一个类统计指标值，然后在对所有类求算术平均值。
            #微平均（Micro-averaging），是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。

            #recall = metrics.recall_score(y_true, y_pred, average='weighted')
            #F1 = metrics.f1_score(y_true, y_pred, average='weighted')
            #acc = metrics.precision_score(y_true, y_pred, average='weighted')
            plot_confusion_matrix(cm_normalized, title= file) 
            # show confusion matrix 
            #plot_confusion_matrix(cm_normalized, title= file) 
            
            plt.savefig(save_dir + '/' + file) 
            #plt.show()
            #print(file)
            #rint(acc,recall)
        except:
            count += 1
            print('this file have problem: ', file)
            #os.remove(re_dir + '/' + file)
    print(count )
        #print(classification_report(y_true, y_pred, target_names=labels))

def make_confusion_matrix_all(re_dir, pr_dir, save_dir):
    #labels表示你不同类别的代号，比如这里的demo中有23个类别 
    #y_true代表真实的label值 y_pred代表预测得到的lavel值 
    y_true, y_pred= get_all_list(re_dir, pr_dir)
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred) 
    np.set_printoptions(precision=2) 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    #print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=150)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array) 
    for x_val, y_val in zip(x.flatten(), y.flatten()): 
        #print(x_val, y_val)
        c = cm_normalized[y_val][x_val] 
        if c > 0.01: 
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7.5, va='center', ha='center')
            # offset the tick 
    plt.gca().set_xticks(tick_marks, minor=True) 
    plt.gca().set_yticks(tick_marks, minor=True) 
    plt.gca().xaxis.set_ticks_position('none') 
    plt.gca().yaxis.set_ticks_position('none') 
    plt.grid(True, which='minor', linestyle='-') 
    plt.gcf().subplots_adjust(bottom=0.15) 
    plt.plot([0,22], [0,22], color="red", linestyle='--', linewidth = 0.5)
    #plt.plot([0,21],[0,21])
    #其中average参数有五种：(None, ‘micro’, ‘macro’, ‘weighted’, ‘samples’)
    #宏平均（Macro-averaging），是先对每一个类统计指标值，然后在对所有类求算术平均值。
    #微平均（Micro-averaging），是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。

    recall1 = '%.3f' % metrics.recall_score(y_true, y_pred, average='macro')
    recall2 = '%.3f' % metrics.recall_score(y_true, y_pred, average='weighted')
    print(recall1, recall2)
    #F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    #acc = metrics.precision_score(y_true, y_pred, average='weighted')
    plot_confusion_matrix(cm_normalized, title= '1-14-200000/' + 'recall1-'+ str(recall1)+'/'+'recall2-'+ str(recall2)) 
    # show confusion matrix 
    #plot_confusion_matrix(cm_normalized, title= file) 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + '1-14-test.png') 


                
if __name__ == "__main__": 
    #re_dir = '/home/hujun/data/diff_mask/mask-test' # 8类的文件位置
    re_dir = '/home/hujun/xiugai61 (复件)'
    pr_dir = '/home/hujun/data/pred-mask/FT/1-14/200000'
    save_dir = '/home/hujun/data/hunxiaojuzheng/1-14'
    labels = ['background(0)','line(1)','road(2)','Traffic sign(3)','person(4)','car(5)',
              'vegetation(6)','sky/building/other(7)', 'road-Shoulder(8)','sidewalk(9)','emergency lane(10)',
             'parking(11)','orbital(12)','traffic sign(13)', 'traffic lights(14)', 'rod(15)',
             'rail(16)', 'person(17)', 'car(18)', 'lawn(19)', 'shrubs(20)', 'other sign(21)', 'sky/building(22)'] 
#     labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'
#               ]
    make_confusion_matrix_all(re_dir, pr_dir, save_dir)
