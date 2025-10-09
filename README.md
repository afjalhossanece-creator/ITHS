# Predicting IT Hardware Service Times and Technicians Performance using Machine Learning

## Project Overview
The main goal of this research is to build a system using ML algorithms and an IT hardware servicing dataset to make the IT servicing process faster and easier. The prepared dataset has been used to train and test ML algorithms (RFC, XGBoost, LightGBM, RF Regressor, and GB Regressor). The training will have two goals (classification and regression).
Finally, the results obtained from both goals have been used together to build a UI that will make the hardware support system modern, accurate, simple, and fast, and it will show our training results (Service Times and Technicians' Performance).


## Technologies Used
**Python:** Core programming language.

**Library (Scikit-learn / Pandas / NumPy/LabelEncoder, etc.):** ML model development & data preprocessing.

**Google Colab:** For developing, training, and testing the machine learning models.

**Gradio:** To create an interactive web-based UI for model inference.

**VS Code:** Used as the primary development environment for building the user interface.

## Installation
First, we will install the following applications/libraries.

* Python 3.8+

* pip

* Google Account (for running notebooks in Colab for model training or testing)

### Install the required dependencies for UI

* VS Code

#### More files need to be added to the UI folder:

1.	app.py

2.	pkl

3.	data file.csv

4.	requirements (gradio, joblib, pandas, numpy, scikit-learn)

#### How to run the Gradio UI

**Step 1:** First, we open the UI folder

**cmd**	        pip install -r requirements.txt

**cmd**	        python app.py

**Step 2:** Then, we will open a web browser and navigate to the local URL given in the terminal (usually http://127.0.0.1:7860). We will use the interface to input the required features, and we will get the prediction results.


## Future Improvements
1 Will try to incorporate deep learning models for more accuracy.

2 Will try to add live integration with ticketing systems.

## Contact
Md: Afjal Hossan Shajal

**ID:** 232217

**Session:** Summer 2023

**Email:** afjalhossan.ece@gmail.com

**GitHub:** https://github.com/afjalhossanece-creator/ITHS.git
