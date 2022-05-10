import matplotlib.pyplot as plt

train_acc = [0.8298, 0.831, 0.8247, 0.8306, 0.8292, 0.8186, 0.8229, 0.8286, 0.826, 0.7802]
test_acc =  [0.828, 0.821, 0.837, 0.838,  0.829, 0.802, 0.844, 0.83, 0.844, 0.767]
lr_list =   [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

l1 = plt.plot(lr_list, train_acc, 'b', label='Train Set')
l2 = plt.plot(lr_list, test_acc, 'y', label='Test Set')
#plt.plot(x1,y1,'ro-',x2,y2,'g+-')
plt.ylim((0.7, 1.0))
plt.xlabel('Learning rate')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.savefig('lr_pre_acc_2.png')
