import os
import json
import numpy as np
import streamlit as st
import onnxruntime as rt
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

scalerSession = rt.InferenceSession("standard_scaler.onnx")
modelSession = rt.InferenceSession("random_forest_model.onnx")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_bmi(weight, height):
    # Calculate BMI
    bmi = weight / (height ** 2)
    
    return bmi

def is_number(text):
    try:
        # Try to convert the text to a float
        float(text)
        return True
    except ValueError:
        # If conversion fails, it's not a number
        return False
    
def diabetic_pedigree_function(mother, father, siblings):
    """
    Calculate a scaled Diabetic Pedigree Function (DPF) for an individual,
    aiming for an output range of approximately (0.078, 2.42).

    Parameters:
    mother (int): 1 if the mother has diabetes, 0 otherwise.
    father (int): 1 if the father has diabetes, 0 otherwise.
    siblings (list): A list of 0s and 1s representing siblings' diabetes status.

    Returns:
    float: The scaled diabetic pedigree function score.
    """
    # Assign weights to each family member
    mother_weight = 0.5
    father_weight = 0.5
    sibling_weight = 0.25
    
    # Calculate the weighted contributions
    family_history = (mother * mother_weight) + (father * father_weight) + (siblings) * sibling_weight
    
    # Add a scaling factor to shift the range
    scaling_factor = 1.2
    bias = 0.078  # Minimum value in the desired range
    
    # Final scaled DPF score
    dpf_score = family_history * scaling_factor + bias
    
    return round(dpf_score, 3)  # Rounded for clarity



if "question_no" not in st.session_state:
    st.session_state.question_no = 2



if st.session_state.question_no == 2:
    progress = st.progress(0)
    age = st.text_input(f"How old are you?", key=st.session_state.question_no, placeholder="Type your answer")
    if st.button("Next->"):
        if not is_number(age):
            st.warning("Please enter a valid age")
        else:
            st.session_state.age = int(age)
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 3:
    progress = st.progress((st.session_state.question_no -2 )/ 8)
    gender = st.selectbox(f"What is your gender?", ["Male", "Female", "Other"], key=st.session_state.question_no)
    if st.button("Next->"):
        st.session_state.gender = gender
        if gender == "Male":
            st.session_state.pregnancies = 0
            st.session_state.question_no += 2
        else:
            st.session_state.question_no += 1
        st.rerun()

elif st.session_state.question_no == 4:
    progress = st.progress((st.session_state.question_no - 2) / 8)
    pregnancies = st.text_input(f"How many times have you been pregnant?", key=st.session_state.question_no)
    if st.button("Next->"):
        if not is_number(pregnancies):
            st.warning("Please enter a vaild Input!!!!")
        else:
            st.session_state.pregnancies = int(pregnancies)
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 5:
    progress = st.progress((st.session_state.question_no -2) / 8)
    glucose = st.text_input(f"Enter your glucose level", key=st.session_state.question_no)
    if st.button("Next->"):
        if not is_number(glucose):
            st.warning("Please enter a valid input!!!!")
        else:
            st.session_state.glucose = int(glucose)
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 6:
    progress = st.progress((st.session_state.question_no-2) / 8)
    bp = st.text_input(f"Enter your blood pressure", key=st.session_state.question_no)
    if st.button("Next->"):
        if not is_number(bp):
            st.warning("Please enter valid input!!!")
        else:
            st.session_state.bp = int(bp)
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 7:
    progress = st.progress((st.session_state.question_no-2) / 8)
    height = st.text_input(f"Enter your height in cm:")
    if st.button("Next->"):
        if not is_number(height):
            st.warning("Please enter valid input!!!!")
        else:
            st.session_state.height = float(height)/100
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 8:
    progress = st.progress((st.session_state.question_no-2) / 8)
    weight = st.text_input(f"Enter your weight in KG")
    if st.button("Next->"):
        if not is_number(weight):
            st.warning("Please enter valid input!!!")
        else:
            st.session_state.weight = float(weight)
            st.session_state.bmi = calculate_bmi(st.session_state.weight, st.session_state.height)
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 9:
    progress = st.progress((st.session_state.question_no-2) / 8)
    st.write("Select the members with diabetes in your family")
    diabeticMother = st.checkbox("Mother")
    diabeticFather = st.checkbox("Father")
    diabeticSibling = st.text_input("Enter the number of diabetic siblings in your family", key=st.session_state.question_no)
    if st.button("Next->"):
        if not is_number(diabeticSibling):
            st.warning("Please enter valid number for diabetic siblings!!!!")
        else:
            st.session_state.diabeticMother = 1 if diabeticMother else 0
            st.session_state.diabeticFather = 1 if diabeticFather else 0
            st.session_state.diabeticSibling = int(diabeticSibling)
            st.session_state.dpf = diabetic_pedigree_function(st.session_state.diabeticMother, st.session_state.diabeticFather, st.session_state.diabeticSibling)
            st.session_state.question_no += 1
            st.rerun()

elif st.session_state.question_no == 10:

    progress = st.progress((st.session_state.question_no-2) / 8)
    input_name = scalerSession.get_inputs()[0].name

    # Transform the new data using the ONNX scaler
    transformed_data = scalerSession.run(None, {input_name: [[st.session_state.pregnancies, st.session_state.glucose, st.session_state.bp, st.session_state.bmi, st.session_state.dpf, st.session_state.age]]})[0]
    # Prepare input data (convert to numpy array)
    input_name = modelSession.get_inputs()[0].name
    test_data = np.array(transformed_data, dtype=np.float32)
    
    # Run the model
    out = modelSession.run(None, {input_name: test_data})[0]
    # out = model.predict([st.session_state.pregnancies, st.session_state.glucose, st.session_state.bp, st.session_state.bmi, st.session_state.dpf])
    out = out[0]

    if out == 0:
        st.markdown("""
            <b>CONGRATS</b> You are not in risk of diabetes type 2
        """, unsafe_allow_html = True)
    else:
        st.markdown("""
            You are in risk of type 2 diabetes. We recommend you to visit doctor as soon as possible.
        """)

    if st.button("Retake test"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()  # Reloads the app to reflect the changes