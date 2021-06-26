# Disaster-Pipeline (Udacity Data Science Nano-degree project)

# Description

The goal of this project is to build a pipeline that correctly classifies messages related to disasters or emergencies. Two data sets "Messages.csv" and "Categories.csv" are combined to form a large data set with over 26,000 entries. After the data is processed it is stored in a SQLite database. The processed messages undergo NLP (Natural Language Processing) so they can be correctly (hopefully) identified as belonging to a certain categorie of an emergency such as fire, weather, earthquake, strorm, etc. There are 36 different categories in total. The project culminizes in a web app where the user can input a message and see if it falls under one of the categories. 

# Visualizations
<img width="431" alt="shot1" src="https://user-images.githubusercontent.com/56938811/123518317-f75ee180-d66a-11eb-99f6-155a7b581170.png">
<img width="947" alt="shot3" src="https://user-images.githubusercontent.com/56938811/123518329-0776c100-d66b-11eb-8f8f-9e195a8e892f.png">
<img width="1050" alt="shot2" src="https://user-images.githubusercontent.com/56938811/123518321-ff1e8600-d66a-11eb-9dc9-85344e532d7a.png">
<img width="1037" alt="shot4" src="https://user-images.githubusercontent.com/56938811/123518356-1a899100-d66b-11eb-9cd1-799895af81c9.png">



# Dependencies 
Python 3
html
sqlalchemy
numpy
panda
sci-kit learn
nltk
pickle
plotly
Flask

# Files

In the 'app' folder:

go.html - code for the web application. Displays the result of what category(s) the message falls under
master.html - Code for displaying visualizations on the web app.
run.py - creates visualizations of the messages data

In the 'data' folder:

disaster_messages.csv - contains data of all the messages to be analyzed by the machine learning model
categories_messages.csv - contains data of all the categories of the messages to be analyzed by the machine learning model
process_data.py - contains the code for all of the data preprocessing of the data sets before being input in to the machine learning pipeline

In the 'models' folder:

train_classifier.py - takes the processed dataframe and tokenizes the messages then trains a classifier with a pipeline and does statistical analysis on it to test the performance of the NLP classifier

# Execution of the Program

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv' data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
   
   In the main 'workspace' directory enter:
    'env|grep WORK
    
  
3. Go to http://WORKSPACEID-3001.WORKSPACEDOMAIN to view and use web application

# Acknowledgements

Thank you to Udacity for mentorship during the project and FigureEight for sharing their dataset

