import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


from streamlit_option_menu import option_menu

import numpy as np
import pickle
import streamlit as st


# loading the saved model
#loaded_model = pickle.load(open('C:/Users/sujee/Downloads/mineproject/trained_model.sav', 'rb'))
loaded_model = pickle.load(open('/path/to/your/trained_model.sav', 'rb'))



def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The person is not diabetic'
    else:
      return'The person is diabetic'
      
def validate_input(value):
    try:
        numeric_value = float(value)
        if numeric_value <= 0:
            st.error("Bhut chhoti expectation hai")
            return False
        elif numeric_value > 10:
            st.error("Bhai/Dost (for girls) shi value enter kre")
            return False
        return True
    except ValueError:
        st.error("Invalid input. Please enter a numeric value.")
        return False

# Load the dataset from the Excel file
#file_path = "C:/Users/sujee/Downloads/mineproject/Book1.xlsx"
file_path = '/path/to/your/Book1.xlsx'
df = pd.read_excel(file_path)

# Extract features (X) and target variable (y)
X = df[['grade', 'softskill', 'problemsolvingskill', 'meditationandyoga', 'discipline', 'strongcommandinoneskill']]
y = df['salaryctc']

# Split the dataset into training and testing sets (you can skip this for the final model)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model and train it on the entire dataset
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict salaryctc for a new dataset
def predict_salary(new_data):
    for column, value in zip(X.columns, new_data):
        if not validate_input(value):
            return None
    new_data_array = np.asarray(new_data, dtype=float)
    new_data_reshaped = new_data_array.reshape(1, -1)
    predicted_salary = model.predict(new_data_reshaped)
    return predicted_salary[0]

# Example usage:
def salaryctc_prediction(input_data):
    # changing the input_data to float
    input_data_as_float = [float(value) for value in input_data]

    # reshape the array as we are predicting for one instance
    predicted_salary = predict_salary(input_data_as_float)
    if predicted_salary is not None:
        return 'Hey you have the potential to earn ' + str(predicted_salary) +'lacs per annum'






      
def main():
    with st.sidebar:
        selected = option_menu('DIABETES and SALARYCTC PREDICTION',
                               ['DIABETES PREDICTION','SALARY PREDICTION','ITS_SU_RJ'],
                               icons=['activity','currency-rupee','emoji-heart-eyes'],
                               default_index=0)
    #diabetes prediction page
    if (selected=='DIABETES PREDICTION'):
        st.title('Diabetes prediction using SVM')

        #getting the input data from user

        Pregnancies = st.text_input('number of Pregnancies')
        Glucose = st.text_input('glucose level')
        BloodPressure = st.text_input('blood pressure value')
        SkinThickness = st.text_input('skin thickness value')
        Insulin = st.text_input('insulin level')
        BMI = st.text_input('bmi value')
        DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction value')
        Age= st.text_input('age value')

        #code for prediction
        diagnosis = ''

        #creating a button for prediction
        if st.button('diabetes test result'):
            diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
            
        st.success(diagnosis)
        
    if (selected=='SALARY PREDICTION'):
        st.title('Salaryctc prediction using multiplelinearregression')
        
        #getting the input data from user
        
        grade = st.text_input('GRADE(1-10)')
        softskill = st.text_input('SOFTSKILL(1-10)')
        problemsolvingskill = st.text_input('PROBLEMSOLVING SKILL(1-10)')
        meditationandyoga = st.text_input('MEDITATION AND YOGA(1-10)')
        discipline = st.text_input('DISCIPLINE LEVEL(1-10)')
        strongcommandinoneskill= st.text_input('STRONG COMMAND IN ONE SKILL(1-10)')
        
        #code for prediction
        diagnosis=''
        
        #creating a button for prediction
        if st.button('SALARY CTC IN LACS'):
            input_data = [grade, softskill, problemsolvingskill, meditationandyoga, discipline, strongcommandinoneskill]
            diagnosis = salaryctc_prediction(input_data)
            
        st.success(diagnosis)
        
    if (selected=='ITS_SU_RJ'):
        st.title('THANK   YOU')
        
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    