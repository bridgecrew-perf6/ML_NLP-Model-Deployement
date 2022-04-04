from flask import Flask,render_template,url_for,request
import nltk
import pandas as pd 
import pickle
import re
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle

# load the model from disk
filename = 'sentimentanalysis_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('bow_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	messages=pd.read_csv("train.csv")
	from nltk.corpus import stopwords
	from nltk.stem.porter import PorterStemmer
	ps = PorterStemmer()
	corpus = []
	for i in range(0, len(messages)):
		review = re.sub('[^a-zA-Z]', ' ', messages['tweet'][i])
		review = review.split()
		review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
		review = ' '.join(review)
		corpus.append(review)
		
		#review = review.lower()
    	
    	
        
        


    	  	

	#Creating the Bag of Words model
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer(max_features=2500)
	X = cv.fit_transform(corpus).toarray()

	y=pd.get_dummies(messages['label'])
	y=y.iloc[:,1].values
	pickle.dump(cv,open('bow_model.pkl','wb') )
	
	# Train Test Split

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
	# Training model using Naive bayes classifier

	from sklearn.naive_bayes import MultinomialNB
	spam_detect_model = MultinomialNB().fit(X_train, y_train)
	pickle.dump(spam_detect_model,open('sentimentanalysis_model.pkl','wb') )
	
    
    

    
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
