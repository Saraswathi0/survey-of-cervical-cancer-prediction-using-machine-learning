from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
#import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

global filename
global df, X_train, X_test, y_train, y_test
global lgb_model

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

import seaborn as sns
def display_graph():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 40))
    
    # Customize color palette
    sns.set_palette("pastel")
    
    # Plot data and customize appearance
    sns.countplot(x='Age', data=df, ax=ax1)
    ax1.set_title('Age Distribution', fontsize=20)  # Increase title font size
    ax1.set_xlabel('Age', fontsize=16)  # Increase x-axis label font size
    ax1.set_ylabel('Count', fontsize=16)  # Increase y-axis label font size
    ax1.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label font size
    ax1.grid(True)  # Add grid lines
    
    sns.countplot(x='Number of sexual partners', data=df, ax=ax2)
    ax2.set_title('Number of Sexual Partners Distribution', fontsize=20)
    ax2.set_xlabel('Number of Sexual Partners', fontsize=16)
    ax2.set_ylabel('Count', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()



def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["Biopsy"], axis=1))
    y = np.array(df["Biopsy"])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")

def adaboost():
    global ada_acc
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for AdaBoost is {ada_acc * 100}%\n'
    text.insert(END, result_text)



def RUN_SVM():
    global svm_acc,svm
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for SVM is {svm_acc * 100}%\n'
    text.insert(END, result_text)



def RF():
    global rf_acc
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Random Forest is {rf_acc * 100}%\n'
    text.insert(END, result_text)


def LR():
    global lr_acc
    from sklearn.linear_model import LogisticRegression
    
    # Initialize logistic regression model
    lr = LogisticRegression()

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Calculate accuracy
    lr_acc = accuracy_score(y_test, y_pred)

    # Display the accuracy
    result_text = f'Accuracy for Logistic Regression is {lr_acc * 100}%\n'
    text.insert(END, result_text)

def plot_bar_graph():
    algorithms = ['AdaBoost', 'SVM', 'Random Forest', 'Logistic Regression']
    accuracies = [ada_acc * 100, svm_acc * 100, rf_acc * 100, lr_acc * 100]
    colors = ['skyblue', 'coral', 'lightgreen', 'lightpink']
    
    plt.figure(figsize=(10, 6))  # Adjust the size of the figure
    bars = plt.bar(algorithms, accuracies, color=colors)

    # Add data labels
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, 
                 bar.get_height() - 3, 
                 f'{accuracy:.2f}%', 
                 ha='center', 
                 color='black', 
                 fontsize=10)

    plt.xlabel('Algorithms', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy of Machine Learning Algorithms', fontsize=14)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.ylim(0, 100)  # Set the y-axis limit from 0 to 100
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


def predict():
    # Open file manager to select CSV file
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if filename:
        # Read the selected CSV file
        input_data = pd.read_csv(filename)

        # Fill missing values with mode for each column
        input_data.fillna(input_data.mode().iloc[0], inplace=True)

        # Preprocess input data (if needed)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])

        # Perform prediction for each row using SVM model
        predictions = svm.predict(input_data)

        # Display the prediction result for each input row
        for idx, prediction in enumerate(predictions):
            if prediction == 1:
                text.insert(END, f"Row {idx + 1}: Cervical Cancer Detected\n")
            else:
                text.insert(END, f"Row {idx + 1}: Cervical Cancer Not Detected\n")

        # Show a message box indicating that prediction is done
        messagebox.showinfo("Prediction Result", "Prediction Completed")


main = tk.Tk()
main.title("Survey of cervical cancer Prediction using Machine Learning: A comparative approach") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = tk.Label(main, text='Survey of cervical cancer Prediction using Machine Learning: A comparative approach',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=600)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=250, y=600)

splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, bg="light green", width=15)
splitButton.place(x=450, y=600)
splitButton.config(font=font1)

adaboostButton = tk.Button(main, text="AdaBoost", command=adaboost, bg="lightgrey", width=15)
adaboostButton.place(x=50, y=650)
adaboostButton.config(font=font1)

RUN_SVM = tk.Button(main, text="RUN_SVM", command=RUN_SVM, bg="pink", width=15)
RUN_SVM.place(x=250, y=650)
RUN_SVM.config(font=font1)

Random_Forest = tk.Button(main, text="Random_Forest", command=RF, bg="yellow", width=15)
Random_Forest.place(x=450, y=650)
Random_Forest.config(font=font1)

Logestic_Regression = tk.Button(main, text="Logestic_Regression", command=LR, bg="lightgreen", width=15)
Logestic_Regression.place(x=650, y=650)
Logestic_Regression.config(font=font1)

graph = tk.Button(main, text="Graph", bg="lightblue", width=15, command=display_graph)
graph.place(x=850, y=650)
graph.config(font=font1)


plotButton = tk.Button(main, text="Plot Results", command=plot_bar_graph, bg="lightgrey", width=15)
plotButton.place(x=50, y=700)
plotButton.config(font=font1)

predict_button = tk.Button(main, text="Prediction", command=predict, bg="orange", width=15)
predict_button.place(x=250, y=700)
predict_button.config(font=font1)

main.config(bg='#32d1a7')
main.mainloop()
