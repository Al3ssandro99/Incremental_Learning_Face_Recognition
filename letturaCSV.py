from csv import  reader
from sklearn.metrics import accuracy_score

csvOut = "outputCSV.csv"  # CSV file path to examinate

with open(csvOut, 'r') as f:
    csv_reader = reader(f)
    local_row = []
    correctName = 0
    totalElaborate = 0
    counter = 0
    y_test = []
    p_test = []

    classes = []
    accuracy = []
    for row in csv_reader:
        y_test.append(row[0])
        p_test.append(row[1])
        totalElaborate += 1
        if local_row == row[0] 
            if row[0] == row[1]:
                correctName += 1
        else:
            counter += 1
            print("CLASSES EXAMINED: " + str(counter))
            classes.append(counter)
            temp = (correctName / (totalElaborate - 1))
            print("ACCURACY: " + str(temp))
            accuracy.append(temp)
            if row[0] == row[1]:
                correctName += 1
        local_row = str(row[0])
    counter += 1
    print("CLASSES EXAMINED: " + str(counter)) # last class
    classes.append(counter)
    temp = (correctName / (totalElaborate - 1))
    print("ACCURACY: " + str(temp))
    accuracy.append(temp)


acc = accuracy_score(y_test, p_test)
print("TOTAL ACCURACY: " + str(acc))

import matplotlib.pyplot as plt


plt.plot(classes, accuracy)
plt.title('ACCURACY GRAPH')
plt.xlabel('Number of classes')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.savefig('result.png')

plt.show()

f= open("resultes.txt", "w+")
f.write("INCREMENTAL LEARNING" + "\n\n")
for i in range (0, len(classes)):
    f.write("CLASSES EXAMINED: " + str(classes[i]) + "\n")
    f.write("ACCURACY: " + str(accuracy[i]) + "\n")
f.write("\n" + "------------------" + "\n")
f.write("TOTAL ACCURACY: " + str(accuracy_score(y_test, p_test)) + "\n")
f.close()
