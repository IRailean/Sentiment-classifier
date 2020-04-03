from ml.model import CharacterCNN
from ml.utils import predict_sentiment
import torch
import os
from tqdm import tqdm
import random
import db
import config
from flask import Flask, Blueprint, request, jsonify

app = Flask(__name__)
api = Blueprint('api', __name__)

model_name = 'model_kaggle_final_1.pth'
model_path = f'./ml/models/{model_name}'

model = CharacterCNN()

if model_name not in os.listdir('./ml/models'):
    print(f'Downloading the trained model {model_name}')
    wget.download('https://github.com/IRailean/Sentiment-classifier/tree/master/api/ml/models/model_kaggle_3_classes.pth', out=model_path)
else:
    print('Model has already been downloaded')
    
# load the model
# set main device GPU/CPU

if torch.cuda.is_available():
    trained_weights = torch.load(model_path)['model']
else:
    trained_weights = torch.load(model_path, map_location='cpu')['model']

model.load_state_dict(trained_weights, strict=True)
model.eval()
print("PyTorch model is set")

# Predict POST
@api.route('/predict', methods=['POST'])
def predict_rating():
    '''
    Endpoint to predict the rating using the
    review's text data.
    '''
    print("In predict_rating method: ")
    if request.method == 'POST':
        if 'review' not in request.form:
            return jsonify({'error': 'no review in body'}), 400
        else:
            parameters = model.get_model_parameters()
            review = request.form['review']
            output = predict_sentiment(model, review, **parameters)
            print("Model output: ", output)
            return jsonify(float(output))
        
# Review POST
@api.route('/review', methods=['POST'])
def post_review():
    '''
    Add review to the database
    '''
    if request.method == 'POST':
        expected_fields = [
            'review',
            'rating',
            'suggested_rating',
            'sentiment_score',
            'brand',
            'user_agent',
            'ip_address'
        ]
        if any(field not in request.form for field in expected_fields):
            return jsonify({'error' : 'Missing field in body'}), 400
        
        query = db.Review.create(**request.form)
        return jsonify(query.serialize())
    
# Reviews GET

@api.route('/reviews', methods=['GET'])
def get_reviews():
    '''
    Get reviews
    '''
    if request.method == 'GET':
        query = db.Review.select()
        return jsonify([q.serialize() for q in query])
    
    
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST)