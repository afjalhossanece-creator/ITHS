import gradio as gr
import pandas as pd
import joblib

# Load trained models
rf_regressor = joblib.load("rf_regressor_checkpoint.pkl")  # regression
lgbm_classifier = joblib.load("lgbm_classifier_checkpoint.pkl")  # classification

# Mappings
solved_by_mapping = {0: 'Moshiur', 1: 'Sarju', 2: 'Shajal', 3: 'Shuvo'}
ui_mapping = {'Female': 0, 'Male': 1}
location_mapping = {
    0: 'ACCL', 1: 'ACML', 2: 'AFL', 3: 'ALP', 4: 'Arkay', 5: 'Azmeri',
    6: 'Badda Parking', 7: 'CHO', 8: 'Cortz', 9: 'Gulshan-29', 10: 'Hamza',
    11: 'KM', 12: 'NKK', 13: 'Nafa', 14: 'Safa'
}
location_mapping = {v: k for k, v in location_mapping.items()}
product_model_mapping = {
    0: 'Acer CPU', 1: 'Acer Laptop', 2: 'Acer Monitor', 3: 'Asus CPU',
    4: 'Asus Laptop', 5: 'Asus Monitor', 6: 'Brother Printer',
    7: 'Canon Printer', 8: 'Dell CPU', 9: 'Dell Laptop', 10: 'Dell Monitor',
    11: 'Epson Printer', 12: 'HP CPU', 13: 'HP Laptop', 14: 'HP Printer',
    15: 'Lenovo CPU', 16: 'Lenovo Laptop', 17: 'Lenovo Monitor',
    18: 'Printer', 19: 'Projector', 20: 'Scanner', 21: 'UPS'
}
product_model_mapping = {v: k for k, v in product_model_mapping.items()}
problem_mapping = {
    0: 'Auto-Restart', 1: 'Battery', 2: 'Broken Issue', 3: 'Color Problem',
    4: 'Display', 5: 'HDD', 6: 'Keyboard/Touchpad', 7: 'Lamp/Temp',
    8: 'Motherboard', 9: 'New Laptop/CPU', 10: 'Not Work', 11: 'OS Problem',
    12: 'Print Problem', 13: 'SSD', 14: 'Slow/Hang Problem'
}
problem_mapping = {v: k for k, v in problem_mapping.items()}
solution_mapping = {'Ready': 0, 'Replace': 1, 'Servicing Issue': 2, 'Warranty Issue': 3}
requisition_mapping = {'No': 0, 'Yes': 1}
speciality_mapping = {'CPU': 0, 'Laptop': 1, 'Monitor': 2, 'Printer/UPS': 3}

# Prediction function
def predict_solver_and_delivery(UI, Location, ProductModel, Problem, Requisition, Solution, Speciality, ReceiveDate):
    try:
        receive_date = pd.to_datetime(ReceiveDate, dayfirst=True, errors="coerce")
        if pd.isna(receive_date):
            return "Error", "Invalid Receive Date format. Use dd-mm-yyyy"

        # Classification input (categorical only)
        X_cls = pd.DataFrame([{
            "UI_E": ui_mapping[UI],
            "Location_E": location_mapping[Location],
            "ProductModel_E": product_model_mapping[ProductModel],
            "Problem_E": problem_mapping[Problem],
            "Solution_E": solution_mapping[Solution],
            "Requisition_E": requisition_mapping[Requisition],
            "Speciality_E": speciality_mapping[Speciality]
        }])

        # Regression input (date only)
        X_reg = pd.DataFrame([{
            "ReceiveYear": receive_date.year,
            "ReceiveMonth": receive_date.month,
            "ReceiveDay": receive_date.day
        }])

        # Predict solver
        cls_pred = lgbm_classifier.predict(X_cls)[0]
        solver = solved_by_mapping.get(cls_pred, "Unknown")

        # Predict delivery days
        reg_pred = rf_regressor.predict(X_reg)[0]
        reg_pred = round(reg_pred)

        return solver, f"{reg_pred} days"
    except Exception as e:
        return "Error", str(e)

# Gradio UI
ui = gr.Interface(
    fn=predict_solver_and_delivery,
    inputs=[
        gr.Radio(list(ui_mapping.keys()), label="UI"),
        gr.Dropdown(list(location_mapping.keys()), label="Location"),
        gr.Dropdown(list(product_model_mapping.keys()), label="Product Model"),
        gr.Dropdown(list(problem_mapping.keys()), label="Problem"),
        gr.Radio(list(requisition_mapping.keys()), label="Requisition"),
        gr.Dropdown(list(solution_mapping.keys()), label="Solution"),
        gr.Dropdown(list(speciality_mapping.keys()), label="Speciality"),
        gr.Textbox(label="Receive Date (dd-mm-yyyy)")
    ],
    outputs=[
        gr.Textbox(label="Predicted Solver"),
        gr.Textbox(label="Predicted Delivery Days")
    ],
    title="Service Request Prediction App",
    description="Predicts the assigned solver and delivery duration (days) from service request details."
)

if __name__ == "__main__":
    ui.launch()
