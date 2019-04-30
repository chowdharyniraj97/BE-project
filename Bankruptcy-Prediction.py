#------------------------------Import libraries--------------------------------------
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from tkinter import *
from tkinter import filedialog
from doit import do_it
#-----------------------end of import-------------------------------



#----------------------------------------------------GUI------------------------
root=Tk();
root.title("Bankruptcy prediction");
root.geometry("400x400+0+0");
path=''

def do_it():
    global path
    path= filedialog.askopenfilename(initialdir="./");

def withdraw(): 
    root.destroy()

heading=Label(root,text="Please select a CSV file", font=("arial",20,"bold"),fg="blue").pack();
select=Button(root,text="Select",width=30,height=2,bg="limegreen",command=do_it).place(x=100,y=75);
#root.withdraw();
close=Button(root,text="close",width=30,height=2,bg="limegreen",command=withdraw).place(x=100,y=150);
root.mainloop();

def popupmsg(msg1):
    popup = Tk()
    popup.wm_title("Accuracy")
    label = Label(popup, text=msg1, font='arail')
    label.pack(side="top", fill="x", pady=10)
    B1 = Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

#---------------------------------end of gui----------------------------------------



#--------------------------LOADING CSV FILES-----------------
filename='dataset/1year.csv'; #csv file used to train the machine
filename2=path                #csv file provided by user through gui
dataframe2=pd.read_csv(filename2)
dataframe=pd.read_csv(filename)
#--------------------------------------------------------------------


#-----------------DATA CLEANING-----------------------------
df=dataframe.drop('id',axis=1)
df2=dataframe2.drop('id',axis=1)
res=df.drop(df.columns[[0,1]], axis=1)
df=res
res2=df

cols=[]
for i in range(65): 
	cols.append('X'+str(i+1))
df.columns=cols
df2.columns=cols
df.rename(columns={'X65':'Y',},inplace=True)
df2.rename(columns={'X65':'Y',},inplace=True)
df=df.replace('?',0)
df2=df2.replace('?',0)
df=df.astype(float)
df.Y=df.Y.astype(int)
df2=df2.astype(float)
df2.Y=df2.Y.astype(int)
df=df.replace(0,np.NaN)
df.Y=df.Y.replace(np.NaN,0)
df.Y=df.Y.astype(int)
df2=df2.replace(0,np.NaN)
df2.Y=df2.Y.replace(np.NaN,0)
df2.Y=df2.Y.astype(int)
#---------------------------------------------------------




#-------------------------DEALING WITH MISSING DATA-----------------
res=pd.DataFrame(fancyimpute.KNN(k=100,verbose=True).fit_transform(df))
res2=pd.DataFrame(fancyimpute.KNN(k=100,verbose=True).fit_transform(df2))
#------------------------------------------------------------


#----------splitting data-----------------------
arrays=df.values
X=arrays[:,0:63]
Y=arrays[:,64]
arraysz=df2.values
X2=arraysz[0:20,0:63]
Y2=arraysz[0:20,64]
#------------------------------------------------------



#-----------------------------------------BALANCING THE DATASET-------------------
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.20,random_state=7)
print(X_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(Y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Y_train==0)))
sm = SMOTE(random_state=2)
X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(Y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res==0)))
#-----------------------------------------------------------


#---------------------------------TRAINING THE MODELS -------------------------------   
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('BB', BalancedBaggingClassifier()))
results = []
names = []
msg=''
for name, model in models:

	cv_results = model_selection.cross_val_score(model, X_train_res, Y_train_res, cv=10, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg+= "%s--> %f \n "  %(name, cv_results.mean()*100)
	
popupmsg(msg)
#-------------------------------------------------------------

#-------------------------------------algo comparision chart--------------------------------
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#-----------------------------------------------------


#---------------------------------TESTING OUR MODEL------------------------------------
balbag= BalancedBaggingClassifier();
balbag.fit(X_train_res, Y_train_res)

predictions = balbag.predict(X_test)
accur="Accuracy of test data:"+str(accuracy_score(Y_test, predictions)*100)
popupmsg(accur)
print(confusion_matrix(Y_test, predictions))


balbag= BalancedBaggingClassifier(RandomForestClassifier());
predictions2=balbag.predict(X2)
popupmsg("Accuracy of unseen data:"+str(accuracy_score(Y2, predictions2)*100))
predictions2=predictions2.astype(int)

output=''
c=1;
for i in predictions2:
    
    if i==1:
       output+="Input "+str(c)+"-->Bankrupt\n"
    else:
        output+="Input "+str(c)+"-->Not Bankrupt\n"
    c=c+1;

popupmsg(output)
#---------------------------------END----------------------------------END---------------------------------------


