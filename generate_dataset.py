import pandas as pd
import numpy as np

np.random.seed(42)
num_records = 5000

# Basic demographic features
age = np.random.randint(18, 70, num_records)
gender = np.random.choice([0,1], num_records)  # 0-Female, 1-Male

# Lifestyle features
diet = np.random.choice([0,1,2], num_records)  # 0-Poor,1-Average,2-Good
activity = np.random.choice([0,1,2], num_records)  # 0-Low,1-Medium,2-High
menstrual_cycle = np.random.choice([0,1], num_records)  # 0-Regular,1-Irregular (only relevant for females)
iron_intake = np.random.choice([0,1,2], num_records)  # 0-Low,1-Medium,2-High
sleep_duration = np.random.choice([0,1,2], num_records)  # 0-Short,1-Normal,2-Long
bmi = np.random.choice([0,1,2,3], num_records)  # 0-Underweight,1-Normal,2-Overweight,3-Obese

# Symptoms (0-No,1-Yes)
pale_skin = np.random.choice([0,1], num_records, p=[0.7,0.3])
cold_hands_legs = np.random.choice([0,1], num_records, p=[0.7,0.3])
weakness = np.random.choice([0,1], num_records, p=[0.6,0.4])
dizziness = np.random.choice([0,1], num_records, p=[0.75,0.25])
short_breath = np.random.choice([0,1], num_records, p=[0.8,0.2])
brittle_nails = np.random.choice([0,1], num_records, p=[0.85,0.15])
sore_tongue = np.random.choice([0,1], num_records, p=[0.9,0.1])
pica = np.random.choice([0,1], num_records, p=[0.95,0.05])
hair_loss = np.random.choice([0,1], num_records, p=[0.8,0.2])
poor_concentration = np.random.choice([0,1], num_records, p=[0.7,0.3])

# Target risk: anemia (simple logic based on poor diet, symptoms, iron intake, menstrual cycle)
risk = ((diet==0) | (pale_skin+weakness+dizziness+hair_loss > 2) |
        (iron_intake==0) | ((gender==0) & (menstrual_cycle==1))).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'diet': diet,
    'activity': activity,
    'menstrual_cycle': menstrual_cycle,
    'iron_intake': iron_intake,
    'sleep_duration': sleep_duration,
    'bmi': bmi,
    'pale_skin': pale_skin,
    'cold_hands_legs': cold_hands_legs,
    'weakness': weakness,
    'dizziness': dizziness,
    'short_breath': short_breath,
    'brittle_nails': brittle_nails,
    'sore_tongue': sore_tongue,
    'pica': pica,
    'hair_loss': hair_loss,
    'poor_concentration': poor_concentration,
    'risk': risk
})

df.to_csv('anemia_extended_5000.csv', index=False)
print("Dataset saved as anemia_extended_5000.csv")
