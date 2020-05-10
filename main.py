from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from keras.models import model_from_json
import cv2
from model import create_model
from Align import AlignDlib
import dlib
import pickle
from PIL import Image
import base64
import io
import keras
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn import preprocessing
from scipy.spatial import distance


app = Flask(__name__)

def get_model():
    global model
    json_file = open('saved/sequential_NN_629_model_output_53dim.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("saved/sequential_NN_629_model_ouput_53dim.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Loaded NN model from disk")
    
def get_openface_model():
	global open_face_model
	open_face_model = create_model()
	open_face_model.load_weights('open_face.h5')
	global graph
	graph = tf.get_default_graph()
	print('Loaded openface model')

    
def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]

def load_names_label_encoder():
	global names_encode
	names_encode = preprocessing.LabelEncoder()
	names_encode.classes_ = np.load('saved/names_encode.npy')

def load_dataset():
	global df
	global names
	global idList
	df = pd.read_csv("embedding.csv")
	names = df['names']
	idList = df['id']
	
def dist(lis1,lis2): 
    s=0
    for x,y in zip(lis1, lis2):
        s=s+((x-y)**2)
    return np.sqrt(s)


def predict_results(path):
	
	image = load_image("uploads/"+path)
	
	faces = alignment.getAllFaceBoundingBoxes(image)
	response = []
	
	for i in range(len(faces)):
		face_aligned = alignment.align(96, image, faces[i], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		face_aligned = (face_aligned / 255.).astype(np.float32)
		
		with graph.as_default():
			embedding = open_face_model.predict(np.expand_dims(face_aligned, axis=0))[0]
		with graph.as_default():
			pred = model.predict([[embedding]])
		ind = np.argsort(pred[0])
		print(ind[::-1][:5])
		prediction=[]
		prediction.append(str(names_encode.inverse_transform([ind[::-1][0]])[0])) 
		prediction.append(str(pred[0][ind[::-1][0]]*100))
		response.append(prediction[0])
		response.append(prediction[1])
		
	return response


def predict_results_ED(path):
	global customerId
	global customerName
	image = load_image("uploads/"+path)
	faces = alignment.getAllFaceBoundingBoxes(image)
	face_aligned = alignment.align(96, image, faces[0], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
	face_aligned = (face_aligned / 255.).astype(np.float32)
	with graph.as_default():
		embedding = open_face_model.predict(np.expand_dims(face_aligned, axis=0))[0]
	original=[]
	for x in embedding:
		original.append(x)
	dist_list=[]
	for idx in range(df.shape[0]):
		temp = []
		for x in df.iloc[idx]:
			temp.append(x)
		dis = dist(original, temp)
		dist_list.append(dis)
	dist_list_idx = sorted(range(len(dist_list)), key=lambda k: dist_list[k])
	response = []
	response.append(names[dist_list_idx[0]])
	customerName = names[dist_list_idx[0]]
	customerId = idList[dist_list_idx[0]]
	response.append(idList[dist_list_idx[0]])
	return response
	
	
	


# Load all preprocessed functions
get_model()
get_openface_model()
load_names_label_encoder()
load_dataset()
alignment = AlignDlib('models/landmarks.dat')

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'satyam'
app.config['MYSQL_DB'] = 'facepay'

# Intialize MySQL
mysql = MySQL(app)

# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests

@app.route('/project/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
	msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
	if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        # Create variables for easy access
		email = request.form['email']
		password = request.form['password']
		accounttype = request.form.get('accounttype')
        # Check if account exists using MySQL
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE email = %s AND password = %s AND accounttype = %s', (email, password, accounttype,))
        # Fetch one record and return result
		account = cursor.fetchone()
		
		if accounttype == "none":
			msg = 'Select Account Type!'
		# If account exists in registration table in out database
		elif account:
            # Create session data, we can access this data in other routes
			session['loggedin'] = True
			session['id'] = account['id']
            # Redirect to home page
			if account['accounttype']=='merchant':
				return redirect(url_for('home'))
			else:
				return redirect(url_for('homeCustomer'))
		else:
            # Account doesnt exist or username/password incorrect
			msg = 'Incorrect username/password/account type!'
    # Show the login form with message (if any)
	return render_template('index.html', msg=msg)


@app.route('/project/pinConfirm', methods=['GET', 'POST'])
def pinConfirm():
	msg=''
	if request.method == 'POST' and 'pin' in request.form:
		pin = request.form['pin']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE pin = %s AND id = %s', (pin, customerId,))
		account = cursor.fetchone()
		if account:
			return render_template('payment.html', account=account)
		else:
			msg='Incorrect PIN'
			return render_template('pinConfirm.html', msg=msg)
	return render_template('pinConfirm.html', msg=msg)
	
	

@app.route('/project/addBalance', methods=['GET', 'POST'])
def addBalance():
	msg=''
	if request.method == 'POST' and 'amount' in request.form:
		amount = int(request.form['amount'])
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session["id"],))
		account = cursor.fetchone()
		prevBalance = int(account["balance"])
		newBalance = prevBalance+amount
		try:
			cursor.execute('UPDATE registration SET balance = %s WHERE id = %s', (newBalance, session["id"],))
			mysql.connection.commit()
			
			#update transaction history
			cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
			cursor.execute('INSERT INTO paymenthistory VALUES (NULL, %s, %s, %s)', (session["id"], session["id"], amount,))
			mysql.connection.commit()
			
			msg='Success'
		except:
			msg='Failed'
		return render_template('addBalance.html', msg=msg)
	return render_template('addBalance.html', msg=msg)	



@app.route('/project/payment', methods=['GET', 'POST'])
def payment():
	msg=''
	if request.method == 'POST' and 'amount' in request.form:
		amount = int(request.form['amount'])
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session["id"],))
		account2 = cursor.fetchone()
		
		#update customer balance
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (customerId,))
		account = cursor.fetchone()
		balanceCustomer = int(account["balance"])
		if balanceCustomer < amount:
			msg='Payment Failed! Insufficient Balance'
			return render_template('acceptPayment.html', msg=msg, account=account2)
			
		newBalanceCustomer = balanceCustomer-amount
		cursor.execute('UPDATE registration SET balance = %s WHERE id = %s', (newBalanceCustomer, customerId,))
		mysql.connection.commit()
		
		#update merchant balance
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session["id"],))
		account1 = cursor.fetchone()
		balanceMerchant = int(account1["balance"])
		newBalanceMerchant = balanceMerchant+amount
		cursor.execute('UPDATE registration SET balance = %s WHERE id = %s', (newBalanceMerchant, session["id"],))
		mysql.connection.commit()
		
		#update transaction history
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('INSERT INTO paymenthistory VALUES (NULL, %s, %s, %s)', (customerId, session["id"], amount,))
		mysql.connection.commit()
		
		msg='Last Transaction Success'
		return render_template('acceptPayment.html', msg=msg, account=account1)
	return render_template('payment.html', msg=msg)		



	
# http://localhost:5000/python/logout - this will be the logout page
@app.route('/project/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
   
   
# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/project/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
	msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'pin' in request.form:
        # Create variables for easy access
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		pin = request.form['pin']
		accounttype = request.form.get('accounttype')
		
		# Check if account exists using MySQL
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE email = %s', (email,))
		account = cursor.fetchone()
        # If account exists show error and validation checks
		if account:
			msg = 'Account already exists!'
		elif accounttype == "none":
			msg = 'Select Account Type!'
		elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address!'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'Username must contain only characters and numbers!'
		elif not username or not password or not email:
			msg = 'Please fill out the form!'
		else:
            # Account doesnt exists and the form data is valid, now insert new account into registration table
			cursor.execute('INSERT INTO registration VALUES (NULL, %s, %s, %s, %s, %s,0)', (username, email, accounttype, password, pin,))
			mysql.connection.commit()
			msg = 'You have successfully registered!'
	elif request.method == 'POST':
        # Form is empty... (no POST data)
		msg = 'Please fill out the form!'
    # Show registration form with message (if any)
	return render_template('register.html', msg=msg)
	
	
# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin merchant users
@app.route('/project/home')
def home():
    # Check if user is loggedin
	if 'loggedin' in session:
        # User is loggedin show them the home page
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session['id'],))
		account = cursor.fetchone()
		if account['accounttype']=='customer':
			return redirect(url_for('homeCustomer'))
		return render_template('home.html', account=account)
    # User is not loggedin redirect to login page
	return redirect(url_for('login'))
	

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin customer users
@app.route('/project/homeCustomer')
def homeCustomer():
    # Check if user is loggedin
	if 'loggedin' in session:
        # User is loggedin show them the home page
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session['id'],))
		account = cursor.fetchone()
		if account['accounttype']=='merchant':
			return redirect(url_for('home'))
		return render_template('homeCustomer.html', account=account)
    # User is not loggedin redirect to login page
	return redirect(url_for('login'))

	
# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/project/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM registration WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
	
	
@app.route('/project/acceptPayment', methods=['GET', 'POST'])
def predict():
	msg = ''
	if request.method == 'POST':
		#image = request.files["image_selector"]
		#image.save(os.path.join("uploads", image.filename))
		image = "image.jpg"
		#name, prob = predict_results(image.filename)
		#name, prob = predict_results(image)
		#msg=name+" "+prob
		name = predict_results_ED(image)
		msg=name
		
		if os.path.exists(os.path.join("uploads", image)):
			os.remove(os.path.join("uploads", image))
			
	if 'loggedin' in session:
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session['id'],))
		account = cursor.fetchone()
		return render_template('acceptPayment.html', msg=msg, account=account)
	return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/captureImage
@app.route('/project/captureImage')
def captureImage():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM registration WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('captureImage.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


@app.route('/project/acceptPayment')
def acceptPayment():
	# Check if user is loggedin
	if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM registration WHERE id = %s', (session['id'],))
		account = cursor.fetchone()
        # Show the profile page with account info
		return render_template('acceptPayment.html', account=account)
    # User is not loggedin redirect to login page
	return redirect(url_for('login'))



@app.route('/project/paymentHistory')
def paymentHistory():
    # Check if user is loggedin
	if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		#(SELECT * FROM paymenthistory WHERE merchantid = %s or customerid= %s)
		cursor.execute('select A.transactionid, custName, merchName, A.amount from (select transactionid,username as custName,amount from (select * from paymenthistory where customerid=%s or merchantid=%s) as C JOIN registration where C.customerid=registration.id) as A JOIN (select transactionid,username as merchName,amount from (select * from paymenthistory where customerid=%s or merchantid=%s) as D JOIN registration where D.merchantid=registration.id) as B where A.transactionid=B.transactionid', (session['id'],session['id'],session['id'],session['id'],))
		account = cursor.fetchall()
        # Show the profile page with account info
		return render_template('paymentHistory.html', output_data=account)
    # User is not loggedin redirect to login page
	return redirect(url_for('login'))
