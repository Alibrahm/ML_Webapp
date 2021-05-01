#Core pkgs
import streamlit as st
st.set_page_config(page_title= 'Nurse Call Data ML App',page_icon='👨‍💻')
#EDA pkgs
import pandas as pd
import numpy as np

#Data visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')

#ML pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy 
from sklearn.metrics import confusion_matrix #to show the confusion matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
import os


from datetime import datetime, timedelta

#utils
import base64
import time
timestr = time.strftime("%Y%m%d.%H%M%S")

#fxtn for  file
def csv_downloader(data):
	csvfile=data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "new_text_file_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href, unsafe_allow_html=True)


def save_uploaded_file(uploadedfile):
	with open(os.path.join("E:/HOLIDAY/New folder/4th Year/Dashboard final GUI/",uploadedfile.name),"wb") as f:
		f.write(uploadedfile.getbuffer())
	return st.success("Saved file: {} in E:/HOLIDAY/New folder/4th Year/Dashboard final GUI".format(uploadedfile.name))

# Class
class FileDownloader(object):
	"""docstring for FileDownloader
	>>> download = FileDownloader(data,filename,file_ext).download()
	"""
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### Download Predictions File ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)





#@st.cache(suppress_st_warning=True) 



#@st.cache(suppress_st_warning=True)  # 👈 Changed this
def main():
    st.title("My Auto Nurse Call Data ML App")
    st.title("IBRAHIM ALI ABDI P15/37272/2016")

    activities = ["EDA","Plot Visualization","Model Building","About Me"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice =='EDA':
       st.subheader("Exploratory Data Analysis")

       data = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
       if data is not None:
        # Making a list of missing value types
        missing_values = ["n/a", "na", "--","0"]
        # Replace using median 

       	df = pd.read_csv(data, na_values = missing_values)
       	#calculates and replaces missing values with median
       	median = df['BloodOxygen'].median()
        df['BloodOxygen'].fillna(median, inplace=True)

        median1 = df['HeartRate'].median()
        df['HeartRate'].fillna(median1, inplace=True)

        median2 = df['BodyTemp'].median()
        df['BodyTemp'].fillna(median2, inplace=True)

       	#Below removes the unmwanted columns in the logged data csv file
       	# to_drop = ['UNIX Timestamp (Milliseconds since 1970-01-01)','Sample Number (100 samples per second)']
       	# df.drop(to_drop, inplace=True, axis=1)
       	st.dataframe(df.head())

        if st.checkbox("Show shape"):
       	    st.write(df.shape)

       	if st.checkbox("Show Columns"):
       	    all_columns = df.columns.to_list()
       	    st.write(all_columns)
 
       	if st.checkbox("Select Columns To Display"):
       		selected_columns = st.multiselect("Select Columns",all_columns)
       		new_df = df[selected_columns]
       		st.dataframe(new_df)

       	if st.checkbox("Show Summary"):
        	st.write(df.describe())

        if st.checkbox("Show Value Counts"):
         	st.write(df.iloc[:,-1].value_counts())


    elif choice =='Plot Visualization':
        st.subheader("Data Visualization")

        data = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
        if data is not None:
       	 # Making a list of missing value types
       	 missing_values = ["n/a", "na", "--","0"]
       	 # Replace using median 
       	 df = pd.read_csv(data, na_values = missing_values)
       	 df.columns = ["SampleNumber","UNIXTimestamp","HeartRate","BloodOxygen","BodyTemp"]
       	 #calculates and replaces missing values with median
       	 median = df['BloodOxygen'].median()
         df['BloodOxygen'].fillna(median, inplace=True)

         median1 = df['HeartRate'].median()
         df['HeartRate'].fillna(median1, inplace=True)

         median2 = df['BodyTemp'].median()
         df['BodyTemp'].fillna(median2, inplace=True)


       	 #Below removes the unmwanted columns in the logged data csv file
       	 #to_drop = ['UNIXTimestamp','SampleNumber']
       	 to_drop = ['UNIXTimestamp','SampleNumber']
       	 #to_drop = ['UNIXTimestamp']
       	 #to_drop1=['SampleNumber']
       	 df.drop(to_drop, inplace=True, axis=1)
       	 st.dataframe(df.head(100))

       	 if st.checkbox("Select to Check Correlation of Features"):
       	 	st.set_option('deprecation.showPyplotGlobalUse', False)
       	 	st.write(sns.heatmap(df.corr(), annot=True,cmap="YlGnBu"))
       	 	st.pyplot()

       	 if st.checkbox("Pie Chart"):
       	 	all_columns = df.columns.to_list()

       	 	columns_to_plot = st.selectbox("Select Only 1 Colummn to See Where Most Activity Occurred",all_columns) 
       	 	pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
       	 	st.write(pie_plot)
       	 	st.pyplot()



       	 all_columns_names = df.columns.tolist()
       	 type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
       	 selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
       	 if st.button("Generate Plot"):
       	 	st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
       	 	if type_of_plot == 'area':
       	 		cust_data = df[selected_columns_names]
       	 		st.area_chart(cust_data)

         	elif type_of_plot == 'bar':
       	 		cust_data = df[selected_columns_names]
       	 		st.bar_chart(cust_data)

       	 	elif type_of_plot == 'line':
       	 		cust_data =  df[selected_columns_names]
       	 		st.line_chart(cust_data)

       	 	elif type_of_plot:
       	 		cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
       	 		st.write(cust_plot)
       	 		st.pyplot()

       	 if st.checkbox("Display line graph"):
       	 	cust_data =  df[selected_columns_names]
       	 	st.line_chart(cust_data)
 







    
    elif choice =='Model Building':
       st.subheader("Building Machine Learning Model")
       data = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
       if data is not None:
       	 # Making a list of missing value types
        missing_values = ["n/a", "na", "--","0"]
        # Replace using median 

       	df = pd.read_csv(data, na_values = missing_values)
       	df.columns = ["SampleNumber","UNIXTimestamp","HeartRate","BloodOxygen","BodyTemp"]
       	#df.columns = ["SampleNumber","UNIXTimestamp","Date","HeartRate","BloodOxygen","BodyTemp"]
       	#df['UNIXTimestamp'] = pd.to_datetime(df['UNIXTimestamp'], unit='ms')

       	#le = preprocessing.LabelEncoder()
       #	df = pd.read_csv(data)
       	#calculates and replaces missing values with median
       	median = df['BloodOxygen'].median()
        df['BloodOxygen'].fillna(median, inplace=True)

        median1 = df['HeartRate'].median()
        df['HeartRate'].fillna(median1, inplace=True)

        median2 = df['BodyTemp'].median()

        df['BodyTemp'].fillna(median2, inplace=True)

        np.nan_to_num(df)
        #df['UNIXTimestamp']=pd.to_datetime(df['UNIXTimestamp'],unit='s')
        #pd.to_datetime(df.UNIXTimestamp,unit="ms")
        #pd.to_datetime(df.UNIXTimestamp,unit="ns")
        # df['Date'] =pd.to_datetime(df['Date'])
        # df['Date'] =df['Date'].dt.strftime('%Y.%m.%d')
        # df['Year'] =pd.DatetimeIndex(df['Date']).year
        # df['Month'] =pd.DatetimeIndex(df['Date']).month
        # df['Day'] =pd.DatetimeIndex(df['Date']).day

        
        














        
        


       	#Below removes the unmwanted columns in the logged data csv file to enable ML
       	# to_drop = ['UNIX Timestamp (Milliseconds since 1970-01-01)','Sample Number (100 samples per second)']
       	# df.drop(to_drop, inplace=True, axis=1)
       	st.dataframe(df.head(100))
       	#df = df.fillna(df.median()).clip(-1e11,1e11)
       	df = df.fillna(2).clip(-1e11,1e11)
       	



       	#np.unique(df)

       	#Building the Model
       	# X = df.iloc[:,0:-1]
       	# Y = df.iloc[:,-1]
       	# seed = 7
       	X = df.drop(["HeartRate"], axis=1) # Features
       	y = df["HeartRate"] # Target variable



       	# Split dataset into training set and test set
       	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

       	# Create Decision Tree classifer object
       	clf = DecisionTreeClassifier()
       	# Train Decision Tree Classifer
       	clf = clf.fit(X_train,y_train)
       	#Predict the response for test dataset
       	y_pred = clf.predict(X_test)

       	

       	#Model
       	# models = []
       	# models.append(("LogisticRegression",LogisticRegression()))
       	# models.append(("LinearDiscriminantAnalysis",LinearDiscriminantAnalysis()))
       	# models.append(("KNeighborsClassifier",KNeighborsClassifier()))
       	# models.append(("DecisionTreeClassifier",DecisionTreeClassifier()))
       	# models.append(("Naive_bayes",GaussianNB()))
       	# models.append(("Support Vector Machine",SVC()))
       	# models = []
       	# models.append(('Logistic Regression', LogisticRegression()))
       	# models.append(('Naive Bayes', GaussianNB()))
       	# models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
       	# models.append(('K-NN', KNeighborsClassifier()))
       	# models.append(('SVM', SVC()))
       	# models.append(('RandomForestClassifier', RandomForestClassifier()))

       	# #evaluate each model in turn

       	# #list
       	# model_names = []
       	# model_mean = []
       	# model_std =[]
       	# all_models = []
       	# scoring = 'accuracy'


       	# # for name,model in models:
       	# # 	kfold = model_selection.KFold(n_splits=10, random_state=None)
       	# # 	cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
       	# # 	model_names.append(name)
       	# # 	model_mean.append(cv_results.mean())
       	# # 	model_std.append(cv_results.std())
       	# for name, model in models:
       	# 	model = model.fit(X_train, y_train)
       	# 	y_pred = model.predict(X_test)
       	# 	from sklearn import metrics

       	# 	accuracy_results = {"model_name":name,"model_accuracy":metrics.accuracy_score(y_test, y_pred)*100}

       	# 	all_models.append(accuracy_results)
       	




       	if st.checkbox("Check Heart Rate Classification Performance Accuracy Metric:"):
       		# st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Standard Deviation"]))
       		st.header("Heart Rate Accuracy Determined using Decision Tree (CART) model supervised Learning")
       		st.write(metrics.accuracy_score(y_test, y_pred)*100)
       		#st.write(confusion_matrix(y_test, y_pred,labels=[1,0]))
       		#sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False)
       		#plt.ylabel("True Values")
       		#plt.xlabel("Predicted Values")
       		#plt.title("CONFUSION MATRIX VISUALIZATION")
       		#plt.show()
       		#st.set_option('deprecation.showPyplotGlobalUse', False)
       		#st.write(sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False))
       		
       		#st.write(sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False))
       		#st.pyplot()


       	if st.checkbox("Heart Rate Classification Performance Report"):
       	 	st.write(classification_report(y_test,y_pred))
       	# 	st.json(all_models)


       	if st.checkbox("Perform HeartRate Predictions"):
       		st.header("Performs Learning and gives the Corresponding Predictions")

       		#my_data = pd.read_csv(data update)
       		#my_file_path ='E:/HOLIDAY/New folder/4th Year/Dashboard final GUI/data update.csv'
       		#my_data = pd.read_csv(my_file_path) 
       		my_data = df.dropna(axis=0)
       		my_data.set_index(df['SampleNumber'],inplace=True)
       		df.drop('SampleNumber',axis=1,inplace=True)
       		y = my_data.HeartRate
       		my_features = ['UNIXTimestamp', 'HeartRate', 'BloodOxygen', 'BodyTemp']
       		X = my_data[my_features]
       		# Define model. Specify a number for random_state to ensure same results each run
       		my_model = DecisionTreeRegressor(random_state=1)
       		# Fit model
       		my_model.fit(X, y)
       		st.text("The Features in the Dataset")
       		my_data.columns
       		my_data = my_data.dropna(axis=0)
       		y = my_data.HeartRate  #the column we want to predict values for
       		my_features = ['HeartRate', 'BloodOxygen', 'BodyTemp']
       		X = my_data[my_features]

       		# Define model. Specify a number for random_state to ensure same results each run
       		my_model = DecisionTreeRegressor(random_state=1)
       		# Fit model
       		my_model.fit(X, y)
       		st.write(DecisionTreeRegressor(random_state=1))

       		st.write("Making predictions for the following 10 Heart Rate readings:")
       		st.write(X.head(10))
       		


       		ids = my_data['SampleNumber']
       		predictions = my_model.predict(X.head(77880)) #the no. of instances in the dataset
       		#output = pd.DataFrame({ 'UNIXTimestamp' : ids, 'HeartRate': predictions })
       		output = pd.DataFrame({ 'SampleNumber' : ids, 'HeartRate': predictions })
       		output.to_csv('HeartRate Now Predictions.csv', index = False)#this successfully downloads the predictions
       		output.to_excel("HeartRate Now Predictions.xlsx")
       		#output.export("E:/HOLIDAY/New folder/4th Year/Dashboard final GUI/HeartRate Predictions.csv", format="wav")
       		output.head()
       		#save_uploaded_file(output.head())
       		#save_uploaded_file(output,'HeartRate Predictions.csv')
       		#st.write(output.head())
       		st.write("The Corresponding HeartRate predictions are as follows:")
       		st.write(output.head(100))
       		#csv_downloader(output.head())
       		download = FileDownloader(output.to_csv(),file_ext='csv').download()



       	if st.checkbox("Check Blood Oxygen Classification Performance Accuracy Metric:"):
       		# st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Standard Deviation"]))
       		X = df.drop(["BloodOxygen"], axis=1) # Features
       		y = df["BloodOxygen"] # Target variable
       		# Split dataset into training set and test set
       		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

       		# Create Decision Tree classifer object
       		clf = DecisionTreeClassifier()
       		# Train Decision Tree Classifer
       		clf = clf.fit(X_train,y_train)
       		#Predict the response for test dataset
       		y_pred = clf.predict(X_test)

       		st.header("Blood Oxygen Classification Accuracy Determined using Decision Tree model supervised Learning")
       		st.write(metrics.accuracy_score(y_test, y_pred)*100)
       		#st.write(confusion_matrix(y_test, y_pred,labels=[1,0]))
       		#sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False)
       		#plt.ylabel("True Values")
       		#plt.xlabel("Predicted Values")
       		#plt.title("CONFUSION MATRIX VISUALIZATION")
       		#plt.show()
       		#st.set_option('deprecation.showPyplotGlobalUse', False)
       		#st.write(sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False))
       		#st.write(sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False))
       		#st.pyplot()
       		

       	if st.checkbox("Perform Blood Oxygen Predictions"):
       		st.header("Performs Learning and gives the Corresponding Predictions")

       		#my_data = pd.read_csv(data update)
       		#my_file_path ='E:/HOLIDAY/New folder/4th Year/Dashboard final GUI/data update.csv'
       		#my_data = pd.read_csv(my_file_path) 
       		my_data = df.dropna(axis=0)
       		# my_data.set_index(df['UNIXTimestamp'],inplace=True)
       		# df.drop('UNIXTimestamp',axis=1,inplace=True)
       		y = my_data.BloodOxygen
       		my_features = ['HeartRate', 'BloodOxygen', 'BodyTemp']
       		X = my_data[my_features]
       		# Define model. Specify a number for random_state to ensure same results each run
       		my_model = DecisionTreeRegressor(random_state=1)
       		# Fit model
       		my_model.fit(X, y)
       		st.text("The Features in the Dataset")
       		my_data.columns
       		my_data = my_data.dropna(axis=0)
       		y = my_data.BloodOxygen  #the column we want to predict values for
       		my_features = ['HeartRate', 'BloodOxygen', 'BodyTemp']
       		X = my_data[my_features]

       		# Define model. Specify a number for random_state to ensure same results each run
       		my_model = DecisionTreeRegressor(random_state=1)
       		# Fit model
       		my_model.fit(X, y)
       		st.write(DecisionTreeRegressor(random_state=1))

       		st.write("Making predictions for the following 10 Blood Oxygen level readings:")
       		st.write(X.head(10))
       		


       		ids = my_data['UNIXTimestamp']
       		predictions = my_model.predict(X.head(77880)) #the no. of instances in the dataset
       		#output = pd.DataFrame({ 'UNIXTimestamp' : ids, 'HeartRate': predictions })
       		output = pd.DataFrame({ 'UNIXTimestamp' : ids, 'HeartRate': predictions })
       		output.to_csv('HeartRate Now Predictions.csv', index = False)#this successfully downloads the predictions
       		output.to_excel("HeartRate Now Predictions.xlsx")
       		#output.export("E:/HOLIDAY/New folder/4th Year/Dashboard final GUI/HeartRate Predictions.csv", format="wav")
       		output.head()
       		#save_uploaded_file(output.head())
       		#save_uploaded_file(output,'HeartRate Predictions.csv')
       		#st.write(output.head())
       		st.write("The Corresponding HeartRate predictions are as follows:")
       		st.write(output.head(100))
       		#csv_downloader(output.head())
       		download = FileDownloader(output.to_csv(),file_ext='csv').download()








    
    elif choice =='About Me':
       st.subheader("About My Project")
       st.header("Nurse Call Sensor Data ML Web Application")
       st.sidebar.header("About App")
       st.sidebar.info("An EDA App for Exploring Receiver Device Data logs")
       st.sidebar.header("Final Year Project")
       #st.sidebar.markdown("[Common ML Dataset Repo]("")")
       st.sidebar.header("Email")
       st.sidebar.info("alibra@students.uonbi.ac.ke")
       st.sidebar.text("Built with Streamlit Python Framework")
       st.sidebar.info("Maintained by IBRAHIM ALI")
       st.balloons()


if __name__=='__main__':
	main()


