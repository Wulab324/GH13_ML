import numpy as np
import matplotlib.pyplot as plt
import itertools as iters

classes=['SH','AS']


classNamber=2 

confusion_matrix = np.array([
    (137,0),
    (0,862),
    ],dtype=np.float64)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

iters = np.reshape([[[i,j] for j in range(classNamber)] for i in range(classNamber)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]),va='center',ha='center')

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
plt.savefig('CM.pdf')
plt.savefig('CM.png',transparent = True)
plt.savefig('CM.svg',format='svg',transparent = True)
plt.show()
