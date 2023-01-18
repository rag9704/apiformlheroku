from fastapi import FastApi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib



app = FastApi()

origins = ["*"]

app.add_middleware(
	CORSMiddleware,
	allow_origins = origins,
	allow_credentials = True,
	allow_methods= ["*"],
	allow_headers = ["*"] 
	)


class mode_input(BaseModel):

	symptom_list = list

	'''
	'itching': int, 
	'skin_rash': int, 
	'nodal_skin_eruptions': int, 
	'continuous_sneezing': int,
    'shivering': int, 
    'chills': int, 
    'joint_pain': int, 
    'stomach_pain': int,
    'acidity': int, 
    'ulcers_on_tongue': int,
    'muscle_wasting': int,
    'vomiting': int,
    'burning_micturition': int,
    'spotting_ urination': int,
    'fatigue': int,
    'weight_gain': int, 
    'anxiety': int, 
    'cold_hands_and_feets': int,
    'mood_swings': int, 
    'weight_loss': int,
    'restlessness': int, 
    'lethargy': int, 
    'patches_in_throat': int, 
    'irregular_sugar_level': int, 
    'cough': int,
    'high_fever': int, 
    'sunken_eyes': int, 
    'breathlessness': int, 
    'sweating': int, 
    'dehydration': int,
    'indigestion': int, 
    'headache': int, 
    'yellowish_skin': int, 
    'dark_urine': int, 
    'nausea': int, 
    'loss_of_appetite': int,
    'pain_behind_the_eyes': int, 
    'back_pain': int, 
    'constipation': int, 
    'abdominal_pain': int, 
    'diarrhoea': int, 
    'mild_fever': int,
    'yellow_urine': int, 
    'yellowing_of_eyes': int, 
    'acute_liver_failure': int, 
    'fluid_overload': int, 
    'swelling_of_stomach': int,
    'swelled_lymph_nodes': int, 
    'malaise': int, 
    'blurred_and_distorted_vision': int, 
    'phlegm': int, 
    'throat_irritation': int,
    'redness_of_eyes': int, 
    'sinus_pressure': int, 
    'runny_nose': int, 
    'congestion': int, 
    'chest_pain': int, 
    'weakness_in_limbs': int,
    'fast_heart_rate': int, 
    'pain_during_bowel_movements': int, 
    'pain_in_anal_region': int,
    'bloody_stool': int,
   	'irritation_in_anus': int, 
   	'neck_pain': int, 
   	'dizziness': int, 
   	'cramps': int, 
   	'bruising': int, 
   	'obesity': int, 
   	'swollen_legs': int,
    'swollen_blood_vessels': int, 
    'puffy_face_and_eyes': int, 
    'enlarged_thyroid': int, 
    'brittle_nails': int, 
    'swollen_extremeties': int,
    'excessive_hunger': int, 
    'extra_marital_contacts': int, 
    'drying_and_tingling_lips': int, 
    'slurred_speech': int,
    'knee_pain': int,
    'hip_joint_pain': int, 
    'muscle_weakness': int, 
    'stiff_neck': int, 
    'swelling_joints': int, 
    'movement_stiffness': int,
    'spinning_movements': int, 
    'loss_of_balance': int, 
    'unsteadiness': int, 
    'weakness_of_one_body_side': int,
    'loss_of_smell': int,
    'bladder_discomfort': int, 
    'foul_smell_of urine': int, 
    'continuous_feel_of_urine': int, 
    'passage_of_gases': int, 
    'internal_itching': int,
    'toxic_look_(typhos)': int, 
    'depression': int, 
    'irritability': int, 
    'muscle_pain': int, 
    'altered_sensorium': int,
    'red_spots_over_body': int, 
    'belly_pain': int, 
    'abnormal_menstruation': int, 
    'dischromic _patches': int, 
    'watering_from_eyes': int,
    'increased_appetite': int, 
    'polyuria': int, 
    'family_history': int,
    'mucoid_sputum': int, 
    'rusty_sputum': int, 
    'lack_of_concentration': int,
    'visual_disturbances': int, 
    'receiving_blood_transfusion': int, 
    'receiving_unsterile_injections': int,
    'coma': int,
    'stomach_bleeding': int, 
    'distention_of_abdomen': int, 
    'history_of_alcohol_consumption': int, 
    'fluid_overload.1': int,
    'blood_in_sputum': int, 
    'prominent_veins_on_calf': int, 
    'palpitations': int, 
    'painful_walking': int, 
    'pus_filled_pimples': int,
    'blackheads': int, 
    'scurring': int, 
    'skin_peeling': int, 
    'silver_like_dusting': int, 
    'small_dents_in_nails': int, 
    'inflammatory_nails': int,
    'blister': int, 
    'red_sore_around_nose': int, 
    'yellow_crust_ooze': int
	'''




model = load(str("decision_tree.joblib"))


@app.post('/prediction')
def predict_disease_from_symptom(input_parameters : model_input):

	input_data = input_parameters

	symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
                'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
                'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
                'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
                'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
                'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
                'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
                'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
                'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
                'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
                'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
                'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
                'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
                'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
                'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}

	for s in input_data:
        symptoms[s] = 1

    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))

    
    result = clf.predict(df_test)
    del df_test

    return f"{result[0]}"

	




