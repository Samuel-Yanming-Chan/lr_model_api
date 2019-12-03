# API Model Deployment

This project is a proof of concept for deploying a model into production.  The end result is a FastAPI app in a docker container which includes the data to train the model.  It can then take in a new data point via the API and return a prediction.

For the purposes of this model the new data point must be labeled data to work but there is already a data point in the testing notebook which is formated correctly.


# Data

The data is taken from the Titanic Disaster data set from Kaggle.
[https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

## Docker

The Dockerfile contains the code to construct a container from the file.  
In order to create the containter navigate to directory in a terminal window and use the **$ docker build .** command.
Since port 3000 was exposed, refer to the Dockerfile, using this run command you can connect it to port 80 on your local machine.
**$ docker run -p80:3000 yourusername/app**

Once that is running you can make an API call to: 
**https://localhost:80**

