# disaster-response-pipeline
### Table of Contents

1. [Installation](#installation)
1.1 [Pre-requirements](#requirement)
1.2 [Data wrangling and model training](#data)
2. [Project Motivation](#maintopic)
3. [File Descriptions](#description)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#license)

## Installation<a name="installation"></a>
### Pre-requirements<a name="requirement"></a>
This code is based on html front-end app, and Python3 back-end which requires several packages and libraries installed.
1. Numpy
2. Pandas
3. Scikit Learn
4. nltk
5. re
6. SQLAlchemy
7. Plotly

All requirements packages details can be found in file requirements.txt
### Data wrangling and model training<a name="data"></a>
In /data, run the following line to generate the database:

`python process_data.py disaster_categories.csv disaster_messages.csv database_filename.db`

After the database is generated, switch to /models, run the following line to train the model:

`python train_classifier.py ../data/database_filename.db clf_model.pkl`

### Render web app
In /app folder, run the following line:

`python run.py`

then, the web app can be open in the host address: http://0.0.0.0:3001/

## Main Topic<a name="maintopic"></a>
In this project, I focus on analyzing the messy and massive messages from different disasters, and building the machine learning model (based on Random Forest Classifier) to classify the messages meaning and categories so that people can use it to predict a new message from social media to identify the categories it should belong to.

## File Description<a name="description"></a>
There are three folders in total.
1. **app** folder, contains the main html files to render: master.html, go.html, run.py
2. **data** folder, contains the raw data and data wrangling scripts: preprocess_data.py
3. **models** folder, contains the machine learning model in pickle file: classifier.pkl
4. License file: LICENCE
5. Readme file: README.md
6. requirement file: requirements.txt

## Preview web app<a name="results"></a>
The rendered web app of this code are available in the following link [here]()
[Preview](/app/static/img/fullscreenshot.png)
## Licensing, Authors, Acknowledgements<a name="license"></a>
This code is run based on the open datasets offered by Figure8(Append) which should be given credit to.

Feel free to use this code for study and sharing thoughts!
