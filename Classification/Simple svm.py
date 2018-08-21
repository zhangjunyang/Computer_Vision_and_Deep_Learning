from sklearn import svm
x = [[2, 0], [1, 1], [3, 3],[2,6]]
y = [0, 0, 1,2]
clf = svm.SVC()
clf.fit(x, y)

print(clf)

# get support vectors
print (clf.support_vectors_)
# get indices of support vectors
print (clf.support_)
# get number of support vectors for each class
print (clf.n_support_)
