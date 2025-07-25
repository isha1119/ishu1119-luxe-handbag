from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = "any_secret_key"  # Needed for session to work

# Load the saved model and encoders
model = joblib.load("svm_brand_model.pkl")  # SVM model
le_sub = joblib.load("subcategory_encoder.pkl")  # Subcategory encoder
label_encoder = joblib.load("brand_label_encoder.pkl")  # Brand encoder

# List of subcategories for dropdown
subcategories = list(le_sub.classes_)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form inputs for user
        subcategory = request.form['subcategory']
        price = float(request.form['price'])
        rating = float(request.form['rating'])

        # Encode subcategory
        sub_encoded = le_sub.transform([subcategory])[0]

        # Form the input array
        user_input = np.array([[sub_encoded, price, rating]])

        # Predict probabilities
        probs = model.predict_proba(user_input)[0]
        predicted_index = np.argmax(probs)
        predicted_brand = label_encoder.inverse_transform([predicted_index])[0]

        # Format probabilities as brand: percent
        brand_probs = {
            label_encoder.inverse_transform([i])[0]: round(p * 100, 2)
            for i, p in enumerate(probs)
        }

        # Store results in session and redirect
        session['prediction'] = predicted_brand
        session['probabilities'] = brand_probs

        return redirect(url_for('thankyou'))

    return render_template("index.html", subcategories=subcategories)

@app.route('/thankyou')
def thankyou():
    prediction = session.get('prediction', 'N/A')
    probabilities = session.get('probabilities', {})
    return render_template("thankyou.html", prediction=prediction, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
