# Food-suggestion
# 🥗 Recipe Recommendation System

This project is a full-stack Machine Learning web application that recommends recipes based on user-inputted nutritional preferences and ingredients. It also filters suggestions based on the desired taste (e.g., sweet, spicy, savory). The app uses a content-based recommendation model built with Python and scikit-learn, and is deployed through a Flask backend with a beautiful HTML/CSS frontend.

---

## 🔍 Features

- 🍛 **Recipe Suggestions**: Get personalized recipe recommendations based on:
  - Calories, Fat, Carbohydrates, Protein, Cholesterol, Sodium, Fiber
  - Ingredients you want to use
  - Desired taste (e.g., sweet, spicy)
- 🧠 **ML-Powered**: Uses TF-IDF vectorization and Nearest Neighbors for similarity-based recommendations.
- 🌐 **Web App Interface**: Clean and modern UI built with Bootstrap, HTML, and CSS.
- 🌙 **Dark Mode** toggle and sidebar menu
- ⏳ **Loading animation** while fetching results

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS, Bootstrap 4
- **Backend**: Python, Flask
- **ML Models**:
  - `TfidfVectorizer` for ingredient text
  - `StandardScaler` for nutrient normalization
  - `NearestNeighbors` for finding similar recipes
- **Libraries**: `scikit-learn`, `numpy`, `pandas`, `matplotlib`

---


