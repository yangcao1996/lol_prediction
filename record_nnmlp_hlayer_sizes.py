import matplotlib.pyplot as plt

train_acc = [0.8266, 0.8242, 0.8189, 0.8274, 0.828, 0.8264, 0.8273, 0.8285, 0.8245, 0.826] 
test_acc =  [0.819, 0.822,  0.842,  00.824, 0.835, 0.823, 0.827, 0.822, 0.815, 0.844]
lr_list =   [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

l1 = plt.plot(lr_list, train_acc, 'b', label='Train Set')
l2 = plt.plot(lr_list, test_acc, 'y', label='Test Set')
plt.ylim((0.7, 1.0))
#plt.plot(x1,y1,'ro-',x2,y2,'g+-')
plt.xlabel('Hidden layer size')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.savefig('layersize_pre_acc_2.png')
