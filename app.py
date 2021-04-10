from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle
import sklearn

app = Flask(__name__, template_folder="templates")
app.static_folder="static"

#infile = open("rain_model1.pkl",'rb')
#new_dict = pickle.load(infile)
model = pickle.load(open("rain_model2.sav", 'rb'))
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/dev.html", methods=['GET'])
@cross_origin()
def dev():
	return render_template("dev.html")

@app.route("/about.html",methods=["GET"])
@cross_origin()
def about():
	return render_template("about.html")

@app.route("/predict",methods=['GET','POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		
		#month
		month=float(request.form['month'])

		# MinTemp
		minTemp = float(request.form['mintemp'])
		# MaxTemp
		maxTemp = float(request.form['maxtemp'])
		# Rainfall
		rainfall = float(request.form['rainfall'])

		
		
		# Wind Gust Speed
		windGustSpeed = float(request.form['windgustspeed'])
		# Wind Speed 9am
		windSpeed9am = float(request.form['windspeed9am'])
		# Wind Speed 3pm
		windSpeed3pm = float(request.form['windspeed3pm'])
		# Humidity 9am
		humidity9am = float(request.form['humidity9am'])
		# Humidity 3pm
		humidity3pm = float(request.form['humidity3pm'])
		# Pressure 9am
		pressure9am = float(request.form['pressure9am'])
		# Pressure 3pm
		pressure3pm = float(request.form['pressure3pm'])
		# Temperature 9am
		temp9am = float(request.form['temp9am'])
		# Temperature 3pm
		temp3pm = float(request.form['temp3pm'])
		
		#location
		location = float(request.form['location'])

		# Rain Today
		rainToday = float(request.form['raintoday'])

		input_cat = [rainToday ,location , month ]
		input_num=[humidity3pm,pressure3pm,windGustSpeed,humidity9am,pressure9am,temp3pm,minTemp
					,maxTemp,temp9am,windSpeed3pm,windSpeed9am,rainfall]
		#normalize karna hai dusre list mai store karna hai fir vo list predict ko pass karna hai
		norm_arr=sklearn.preprocessing.normalize([input_num])
		
		arr1=np.insert(norm_arr,2,rainToday)
		arr2=np.insert(arr1,9,location)
		arr3=np.insert(arr2,13,month)
		
		

				 
		#
		pred = model.predict([arr3])
		output = pred
		if output == 0:
			return render_template("sunny.html")
		else:
			return render_template("rainy.html")
	return render_template("predict.html")
		
if __name__=='__main__':
	app.run(debug=True)			 

	
