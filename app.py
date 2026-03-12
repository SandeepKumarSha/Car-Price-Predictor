from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset
car = pd.read_csv("Cleaned Car.csv")

# Load trained model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# Dropdown values
companies = sorted(car['company'].unique())
fuel_types = car['fuel_type'].unique()
years = sorted(car['year'].unique(), reverse=True)

# ---> NEW: Create a dictionary matching each company to its list of models
company_models_dict = {}
for comp in companies:
    # Filter the dataset by company and get unique models
    company_models_dict[comp] = sorted(car[car['company'] == comp]['name'].unique().tolist())

@app.route('/')
def index():
    return render_template(
        'index.html',
        companies=companies,
        company_models_dict=company_models_dict, # Passed to frontend
        fuel_types=fuel_types,
        years=years
    )

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    company = request.form.get('company')
    year = int(request.form.get('year'))
    kms = int(request.form.get('kms'))
    fuel = request.form.get('fuel')

    input_df = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=['name','company','year','kms_driven','fuel_type']
    )

    prediction = model.predict(input_df)

    return render_template(
        'index.html',
        prediction=int(prediction[0]),
        companies=companies,
        company_models_dict=company_models_dict, # Passed to frontend here too
        fuel_types=fuel_types,
        years=years,
        selected_company=company,
        selected_name=name,
        selected_year=year,
        selected_fuel=fuel,
        selected_kms=kms
    )

if __name__ == "__main__":
    app.run(debug=True)
