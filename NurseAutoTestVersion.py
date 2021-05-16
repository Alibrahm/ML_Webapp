#Core pkgs
import streamlit as st
st.set_page_config(page_title= 'Nurse Call Data ML App',page_icon='👨‍💻',layout = 'wide')
#EDA pkgs
import pandas as pd
import numpy as np

#Data visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
#%matplotlib inline

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

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
import itertools
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
from IPython.display import HTML

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
    #st.title("My Auto Nurse Call Data ML App")
    #st.title("IBRAHIM ALI ABDI P15/37272/2016")
    st.markdown("<h1 style='text-align: center; color: red;'>My Auto Nurse Call Data Analytics Tool</h1>", unsafe_allow_html=True)

    activities = ["EDA","Plot Visualization","Model Building","About Me","Help"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if st.checkbox("Help"):
       #st.header("Email")
       st.markdown("<h3 style='text-align: center; color: blue;'>How to Use</h3>", unsafe_allow_html=True)
       #st.info("alibra@students.uonbi.ac.ke")
       st.text("In the upper left corner click on the '>' symbol to pull the sidebar and select analytics activities ")
       st.text("Below you should first import the required dataset that was logged by desktop application real-time monitor ")
       st.text("Maintained by IBRAHIM ALI. You can reach me through ✅alibra@students.uonbi.ac.ke or alibra71@gmail.com")

    if choice =='EDA':
       
       #st.subheader("Exploratory Data Analysis")
       st.markdown("<h3 style='text-align: center; color: yellow;'>Exploratory Data Analysis</h3>", unsafe_allow_html=True)

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
        st.markdown("<h6 style='text-align: center; color: yellow;'>Below is Dataset you imported</h6>", unsafe_allow_html=True)
       	st.dataframe(df.head())

        #st.markdown("## Exploring The Dataset Above")
        st.markdown("<h3 style='text-align: center; color: yellow;'>Exploring The Dataset Above</h3>", unsafe_allow_html=True)

        # kpi 1 
        kpi1, kpi2 = st.beta_columns(2)
        with kpi1:
          #st.markdown("**Features**")
          st.markdown("<h4 style='text-align: center; color: yellow;'>Shape of Data</h4>", unsafe_allow_html=True)
          st.write(df.shape)
          st.info("Above values shows the number of rows,columns in the Dataset ")
          #st.info("rows,columns in the Dataset") 

          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

        with kpi2:
          #st.markdown("**Attributes**")
          st.markdown("<h4 style='text-align: center; color: yellow;'>Attributes/Columns in Dataset</h4>", unsafe_allow_html=True)
          all_columns = df.columns.to_list()
          st.write(all_columns)
          st.info("Above shows the attributes/features in the Dataset ")
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

        

          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)


        #st.markdown("## Select Features of interest To Display")
        st.markdown("<h2 style='text-align: center; color: yellow;'>Select Features of interest To Display</h2>", unsafe_allow_html=True)
        selected_columns = st.multiselect("Select Columns you want to check for Correlations",all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

        #st.markdown("<hr/>",unsafe_allow_html=True)


        st.markdown("<hr/>",unsafe_allow_html=True) #for the next row 


        #st.markdown("## Select Features of interest To Display")
        st.markdown("<h2 style='text-align: center; color: yellow;'>Below is the Summary Statistics of the Dataset</h2>", unsafe_allow_html=True)
        st.info("Statistics such as mean, standard deviations and quartiles of where most instances are distributed in the dataset logged by the wearable are as below")
        st.write(df.describe())


        st.markdown("<hr/>",unsafe_allow_html=True)

        


    elif choice =='Plot Visualization':
        st.subheader("Data Visualization")

        datas = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
        if datas is not None:
       	 # Replace using median 
       	 df = pd.read_csv(datas)
       	 df.columns = ["SampleNumber","UNIXTimestamp","RecordedInstanceTime","HeartRate","BloodOxygen","BodyTemp"]
         #df.index = pd.DatetimeIndex(df.index).to_period('S')
       	 #calculates and replaces missing values with median
       	 median = df['BloodOxygen'].median()
         df['BloodOxygen'].fillna(median, inplace=True)

         median1 = df['HeartRate'].median()
         df['HeartRate'].fillna(median1, inplace=True)

         median2 = df['BodyTemp'].median()
         df['BodyTemp'].fillna(median2, inplace=True)


       	 #Below removes the unmwanted columns in the logged data csv file
       	 #to_drop = ['UNIXTimestamp','SampleNumber']
       	 to_drop = ['UNIXTimestamp','SampleNumber','RecordedInstanceTime']
       	 #to_drop = ['UNIXTimestamp']
       	 #to_drop1=['SampleNumber']
       	 df.drop(to_drop, inplace=True, axis=1)
       	 st.dataframe(df.head(100))


         #st.markdown("## KPI First Row")
         st.markdown("<h2 style='text-align: center; color: yellow;'>Key Performance Indicators</h2>", unsafe_allow_html=True)
         st.markdown("<p style='text-align: center; color: white;'>Below shows how features are related and distributed.<br>Hover above the charts and select the expand button to zoom into the charts</p>", unsafe_allow_html=True)
         # kpi 1 
         kpi1, kpi2, kpi3 = st.beta_columns(3)
         with kpi1:
          #st.markdown("**Correlation KPI**")
          st.markdown("<h4 style='text-align: center; color: White;'>Feature Correlation KPI</h4>", unsafe_allow_html=True)
          #number1 = 111 
          st.set_option('deprecation.showPyplotGlobalUse', False)
          st.write(sns.heatmap(df.corr(), annot=True,cmap="YlGnBu"))
          st.pyplot()          
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

        with kpi2:
          #st.markdown("**Second KPI**")
          st.markdown("<h4 style='text-align: center; color: White;'>Feature Distribution KPI</h4>", unsafe_allow_html=True)
          number2 = 222 
          all_columns = df.columns.to_list()
          columns_to_plot = st.selectbox("Select Only 1 Target Column to Render Where Most Activity Occurred",all_columns)
          pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%.f%%")
         
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)


        with kpi3:
          #st.markdown("**Plot Selected KPI**")
          st.markdown("<h4 style='text-align: center; color: White;'>Plot of Feature Distribution KPI</h4>", unsafe_allow_html=True)
          number3 = 333 
          st.write(pie_plot)
          st.pyplot()           
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center; color: yellow;'>Custom Visual Render Charts</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white;'>Please select your chart of choice.<br>You can only interact with one chart at a time as rendering all charts simultaneously consumes heavy computer memory</p>", unsafe_allow_html=True)
        #datac = st.beta_columns(1)
        #with datac:
        number2 = 222
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Chart Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
        if st.button("Generate Plot"):
          st.success("I'm now Generating your Customizable Plot of type {} for {}".format(type_of_plot,selected_columns_names))

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

        
        #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)






 







    
    elif choice =='Model Building':
       st.subheader("Building Machine Learning Model")
       datas = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
       if datas is not None:
       	 # Making a list of missing value types
        #missing_values = ["n/a", "na", "--","0"]
        # Replace using median 

       	data = pd.read_csv(datas)
        #data['RecordedInstanceTime'] = pd.DatetimeIndex(data['RecordedInstanceTime']).to_period('S')
        #data.set_index(data['RecordedInstanceTime'],inplace=True)
        #data.drop('RecordedInstanceTime',axis=1,inplace=True)

        data = data.dropna(axis=0)
        data.set_index(data['RecordedInstanceTime'],inplace=True)
        data.drop('RecordedInstanceTime',axis=1,inplace=True)
       	data.columns = ["SampleNumber","UNIXTimestamp","HeartRate","BloodOxygen","BodyTemp"]

        #median = data['BloodOxygen'].median(
        #data['BloodOxygen'].fillna(median, inplace=True)
        #data['HeartRate'].fillna(method='ffill', inplace=True)
        data['HeartRate']=data['HeartRate'].replace(0,data['HeartRate'].mode()).astype(int)
        #data['HeartRate']=data['HeartRate'].replace(2,data['HeartRate'].mean()).astype(int) #to replace value 2
        data['HeartRate']=data['HeartRate'].replace(200,data['HeartRate'].median()).astype(int) #to replace value 200
        data['HeartRate']=data['HeartRate'].replace(204,data['HeartRate'].median()).astype(int) #to replace value 204
        data['HeartRate']=data['HeartRate'].replace(179,data['HeartRate'].mean()).astype(int)#to replace value 179 
        data['HeartRate']=data['HeartRate'].replace(134,data['HeartRate'].mean()).astype(int)
        #data['HeartRate']=data['HeartRate'].replace(6,data['HeartRate'].mean()).astype(int) #to replace value 2
        data['BloodOxygen']=data['BloodOxygen'].replace(0,data['BloodOxygen'].mode()).astype(int)
        data['BodyTemp']=data['BodyTemp'].replace(0,data['BodyTemp'].mean()).astype(int)

        to_drop = ['UNIXTimestamp']
        data.drop(to_drop, inplace=True, axis=1)
        to_drop1 = ['SampleNumber']
        data.drop(to_drop1, inplace=True, axis=1)
        to_drop2 = ['BloodOxygen']
        to_drop3 = ['BodyTemp']
        data.drop(to_drop2, inplace=True, axis=1)
        data.drop(to_drop3, inplace=True, axis=1)


        st.markdown("## The Target Feature To perform Machine Learning Predictions") 
        st.dataframe(data.head())
        st.markdown("<hr/>",unsafe_allow_html=True)



        


        st.markdown("## KPI First Row") 
        kpi1, kpi2, kpi3 = st.beta_columns(3)
        with kpi1:
          st.markdown("**First KPI**")
          number1 = st.dataframe(data.head(100)) 
          st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

        with kpi2:
          st.markdown("**Second KPI**")
          number2 = 222 
          st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

        with kpi3:
          st.markdown("**Third KPI**")
          number3 = 333 
          st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)




        








    
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


