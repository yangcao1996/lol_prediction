import matplotlib.pyplot as plt

train_acc = [0.826, 0.8287, 0.8284, 0.8308, 0.83]
test_acc =  [0.844, 0.828,  0.814,  0.823,  0.831]
lr_list =   [1, 2, 3, 4, 5]

l1 = plt.plot(lr_list, train_acc, 'b', label='Train Set')
l2 = plt.plot(lr_list, test_acc, 'y', label='Test Set')
#plt.plot(x1,y1,'ro-',x2,y2,'g+-')
plt.ylim((0.7, 1.0))
plt.xlabel('Hidden layer number(depth)')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.savefig('layernumber_pre_acc_2.png')
