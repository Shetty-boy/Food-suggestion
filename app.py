from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ast
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import normalize
from time import time
from scipy.spatial.distance import cosine, euclidean, hamming

app = Flask(__name__)

dff=pd.read_csv('new_dataset_with_taste.csv')

vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(dff['ingredients_list'])

scaler = StandardScaler()
X_numerical = scaler.fit_transform(dff[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber',]])
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)


def recommend_recipes(input_features):
    input_features_scaled = scaler.transform([input_features[:7]])
    input_ingredients_transformed = vectorizer.transform([input_features[7]])
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
    distances, indices = knn.kneighbors(input_combined)
    recommendations = dff.iloc[indices[0]]
    result_df = dff[dff['recipe_name'].isin(recommendations['recipe_name'])]
    return result_df[['recipe_name', 'ingredients_list','calories','fat','protein','image_url','predicted_taste']]

def finalrecommendrecipe(recc, taste):

    if isinstance(recc, list):
        recc = pd.DataFrame(recc)


    recc['predicted_taste'] = recc['predicted_taste'].str.strip().str.lower()
    taste = taste.strip().lower()

    result_df = recc[recc['predicted_taste'] == taste]


    if not result_df.empty:
        return result_df[['recipe_name', 'ingredients_list', 'calories', 'fat', 'protein', 'image_url', 'predicted_taste']]
    else:
        return recc
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']
        taste=request.form['taste']
        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = recommend_recipes(input_features)
        recc=finalrecommendrecipe(recommendations,taste)
        return render_template('index.html', recommendation=recc.to_dict(orient='records'),truncate = truncate)
    return render_template('index.html', recommendation=[])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)