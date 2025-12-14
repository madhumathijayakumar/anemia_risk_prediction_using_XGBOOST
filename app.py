from flask import Flask, request, render_template
import pickle
import numpy as np
import shap

app = Flask(__name__)

# Load trained XGBoost model
with open('anemia_model_extended.pkl', 'rb') as f:
    model = pickle.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Function to generate diet recommendations
def get_diet_recommendation(data):
    recommendations = []

    if data['diet'] == 0:
        recommendations.append("Your diet seems poor. Include iron-rich foods like spinach, legumes, eggs, red meat, and fortified cereals.")
    elif data['diet'] == 1:
        recommendations.append("Your diet is average. Try to include more iron-rich foods regularly.")

    if data['iron_intake'] == 0:
        recommendations.append("Iron intake is low. Consider iron-rich foods or supplements if recommended by your doctor.")
    elif data['iron_intake'] == 2:
        recommendations.append("Good iron intake. Maintain it.")

    if data['bmi'] == 0:
        recommendations.append("You are underweight. Include more protein and calories in your diet.")
    elif data['bmi'] == 2 or data['bmi'] == 3:
        recommendations.append("Maintain a balanced diet to manage weight.")

    if data['menstrual_cycle'] == 1:
        recommendations.append("Irregular menstrual cycle detected. Ensure sufficient iron intake and consult a doctor if needed.")

    if data['sleep_duration'] == 0:
        recommendations.append("Short sleep may affect health and iron absorption. Aim for 6-8 hours of sleep.")

    # Symptoms
    symptom_list = ['pale_skin','cold_hands_legs','weakness','dizziness','short_breath','brittle_nails','sore_tongue','pica','hair_loss','poor_concentration']
    symptoms_present = [sym.replace("_"," ") for sym in symptom_list if data[sym]==1]
    if symptoms_present:
        recommendations.append("You have symptoms like " + ", ".join(symptoms_present) + ". Consider nutrient-rich foods (iron, B12, folate, protein).")

    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features from form
        features = np.array([[float(request.form['age']),
                              int(request.form['gender']),
                              int(request.form['diet']),
                              int(request.form['activity']),
                              int(request.form['menstrual_cycle']),
                              int(request.form['iron_intake']),
                              int(request.form['sleep_duration']),
                              int(request.form['bmi']),
                              int(request.form.get('pale_skin',0)),
                              int(request.form.get('cold_hands_legs',0)),
                              int(request.form.get('weakness',0)),
                              int(request.form.get('dizziness',0)),
                              int(request.form.get('short_breath',0)),
                              int(request.form.get('brittle_nails',0)),
                              int(request.form.get('sore_tongue',0)),
                              int(request.form.get('pica',0)),
                              int(request.form.get('hair_loss',0)),
                              int(request.form.get('poor_concentration',0))]])

        # Prediction
        prediction = model.predict(features)[0]
        result = "At Risk of Anemia" if prediction==1 else "Not at Risk"

        # SHAP explanation
        shap_values = explainer.shap_values(features)
        feature_names = ['age','gender','diet','activity','menstrual_cycle','iron_intake','sleep_duration','bmi',
                         'pale_skin','cold_hands_legs','weakness','dizziness','short_breath','brittle_nails',
                         'sore_tongue','pica','hair_loss','poor_concentration']
        contributions = dict(zip(feature_names, shap_values[0]))
        top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_text = ", ".join([f"{f[0]} ({'increases' if f[1]>0 else 'decreases'} risk by {abs(f[1]):.2f})" 
                              for f in top_features])

        # Prepare data dict for diet recommendations
        input_data = {
            'age': float(request.form['age']),
            'gender': int(request.form['gender']),
            'diet': int(request.form['diet']),
            'activity': int(request.form['activity']),
            'menstrual_cycle': int(request.form['menstrual_cycle']),
            'iron_intake': int(request.form['iron_intake']),
            'sleep_duration': int(request.form['sleep_duration']),
            'bmi': int(request.form['bmi']),
            'pale_skin': int(request.form.get('pale_skin',0)),
            'cold_hands_legs': int(request.form.get('cold_hands_legs',0)),
            'weakness': int(request.form.get('weakness',0)),
            'dizziness': int(request.form.get('dizziness',0)),
            'short_breath': int(request.form.get('short_breath',0)),
            'brittle_nails': int(request.form.get('brittle_nails',0)),
            'sore_tongue': int(request.form.get('sore_tongue',0)),
            'pica': int(request.form.get('pica',0)),
            'hair_loss': int(request.form.get('hair_loss',0)),
            'poor_concentration': int(request.form.get('poor_concentration',0))
        }

        recommendations = get_diet_recommendation(input_data)

        return render_template('result.html', prediction=result, explanation=top_text, recommendations=recommendations)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
