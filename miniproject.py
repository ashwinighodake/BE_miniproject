#from tkinter import *
import matplotlib.pyplot as plt 
#pyplot is matplotlib's plotting framework. That specific import line merely imports the module "matplotlib.pyplot" and binds that to the name "plt"#
import matplotlib as rc1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import f2 as slct
#from tkinter import *
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.externals import joblib 
from sklearn.datasets.samples_generator import make_blobs
from tkinter import *
from tkinter.messagebox import *
font={'family':'normal','weight':'bold','size':6}
rc1.rc('font',**font)

def open_window1():
    top = Toplevel()
    top.title("Crop Analysis")
    top.geometry("700x600+120+120")
    #top.configure(background = '#34baeb')
    pf=pd.read_csv('demo.csv')
    #GUI
    options=list(pf['DISTRICT'])
    master=Tk()
    #master.geometry('700x600')
    text=Label(top, text=' CROP Analysis',)
    selectoption=StringVar()
    selectoption.set(options[0])#default
    
    optionmenu=OptionMenu(top,selectoption,*options)
    
    #button=Button(top,text='Search',command=graph)
    #button.pack()
    text.pack()
    optionmenu.pack()
    #crime graph

def graph():
    op=selectoption.get()
    indx=slct.f2(op)
    x_label=['Iron', 'Calcium','Magnesium','Potassium']
    s=pf.iloc[indx,0:5]
    p=list(s)
    freq=p[1:]
    xs = np.arange(len(freq))
    plt.bar(xs,freq)
    plt.ylabel('Rate')
    plt.xlabel('Amount')
    plt.xticks(xs, x_label,rotation='vertical')
    plt.title(op)
    plt.show()

button=Button(top,text='Search',command=graph)
#button.place(x=330,y=60)
button.pack()
''' c=pf.iloc[0:45,1]

  label=list(pf['DISTRICT'])
  data=list(c)
  x=np.arange(len(data))
  plt.bar(x,data)
  plt.ylabel('RATE')
  plt.xticks(x, label,rotation='vertical')
  plt.title('IRON required \n 2013')
  plt.show()
'''
def open_window2():
    top=Toplevel()
    top.title("CROP PREDICTION")
    top.geometry("700x600+120+120")
    top.configure(background='spring green')
    l1=Label(top, text="iron")
    l1.grid(row=2,column=0)
    l2=Label(top, text="calcium")
    l2.grid(row=3,column=0)
    l3=Label(top, text="magnesium")
    l3.grid(row=4,column=0)
    l4=Label(top, text="Potassium")
    l4.grid(row=5,column=0)
    l5=Label(top, text="Mineral")
    l5.grid(row=6,column=0)
    l6=Label(top, text=" Best Suitable Crop for your Farm")
    l6.grid(row=21,column=1)
    Label(top, text = "Output").grid(row=30) 
    #blank = Entry(window)
    blank = Entry(top, width=35, bg="white")
    blank.grid(row=30, column=1)
    iron=StringVar()
    e1=Text(top, width=25, height=3, bg="white")
    e1.grid(row=2,column=1)
    calcium=StringVar()
    e2=Text(top, width=25, height=3, bg="white")
    e2.grid(row=3,column=1)
    magnesium=StringVar()
    e3=Text(top, width=25, height=3, bg="white")
    e3.grid(row=4,column=1)
    equine=StringVar()
    e4=Text(top, width=25, height=3, bg="white")
    e4.grid(row=5,column=1)
    crop=StringVar()
    e5=Text(top, width=25, height=3, bg="white")
    e5.grid(row=6,column=1)

def process_data():
    iron = e1.get("1.0","end-1c")
    num_iron = int(float(iron))
    calcium = e2.get("1.0","end-1c")
    num_calcium = int(float(calcium))
    magnesium = e3.get("1.0","end-1c")
    num_mag = int(float(magnesium))
    equine = e4.get("1.0","end-1c")
    num_equine = int(float(equine))
    crop = e5.get("1.0","end-1c")
    num_crop = int(float(crop))
    names = ['iron', 'calcium', 'magnesium', 'quine', 'crop','fertile']
    dataframe = pandas.read_csv("a1_2.csv")
    array = dataframe.values
    X = array[:,0:5]
    Y = array[:,5]
    test_size = 0.33
    seed = 5
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size,random_state=seed)
    # Fit the model on 33%
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # save the model to disk
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)

    # load the model from disk
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X_test, Y_test)
    print(result)
    X, y = make_blobs(n_samples=100, centers=2, n_features=2,random_state=1)
    Xnew = [[num_iron, num_calcium, num_mag, num_equine, num_crop]]
    ynew = model.predict(Xnew)
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
    Ans = ynew[0]
    blank.insert(0, Ans)
    #enter = Button(root, text="Enter Data", width=30, height=5,bg="lightblue", command=enter_data).place(x=250, y=300)

    #process=Button(window, text="Process Data", width=30, height=5,bg="lightblue", command=process_data).place(x=500, y=300)

    process=Button(top,text="Submit",width=12,command=process_data)
    process.grid(row=12,column=1)
    
root = Tk()
frame = Frame(root)
root.title("CROP PREDITION USING SOIL PARAMETERS")
root.geometry('700x600+120+120')
frame1 = Frame(root, height = 150, bg = 'white')
frame2 = Frame(root, height = 500, bg = '#34baeb')
heading = Label(root, text = 'Analysis and Prediction of crop', font ='arial 15 bold',bg = 'white',fg = '#ebb434')
heading.place(x = 200, y = 50)
button1 = Button(root,text = " Wheat Crop Analysis ",bg = 'white',fg ='#ebb434', font = 'arial 12 bold',command = open_window1)
button1.place(x=250,y=200)
button2 = Button(root,text = "Crop Prediction",bg = 'white',fg ='#ebb434',font = 'arial 12 bold',command = open_window2)
button2.place(x=250,y=250)
frame1.pack(fill=X)
frame2.pack(fill=X)
root.mainloop()

'''
root = Tk()
#app = Application(root)
root.title("CROP ANALYSIS
AND PREDCTION USING SOIL PARAMETERS")
#canvas = Canvas(width = '1000',height = '1000')
#canvas.pack()#photo = PhotoImage(file='C:\\Python Project\\image.png')
#canvas.create_image(0, 0, anchor = NW, image = photo)
root.geometry("700x600+120+120")
root.configure(background='black')
#frame = Frame(root)button1 = Button(root, text = "Crop Analysis",bg =
"white",command=open_window1)
button2 = Button(root, text = "Crop Predcition", bg =
"white",command=open_window2)
#button1.grid(row = 0,column = 3, padx = 100)
#button2.grid(row = 1,column = 3, pady = 10, padx = 20)
button1.pack()
button2.pack()
#frame.pack()
root.mainloop()
'''