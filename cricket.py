import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the app
st.title("Cricket Match Winner Predictor")
st.write("Predict the winner of cricket matches based on historical data.")

# Load the CSV file using st.cache_data to cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv('ODI_Match_info.csv')
    df_filtered = df[['team1', 'team2', 'venue', 'winner']].dropna()  # Filter necessary columns and drop rows with missing 'winner'
    return df_filtered

# Load the data
df = load_data()

# Display the dataset
st.write("Dataset Preview:")
st.write(df.head())

# Feature encoding with LabelEncoder
label_encoder_team = LabelEncoder()
label_encoder_venue = LabelEncoder()
label_encoder_winner = LabelEncoder()

df['team1_encoded'] = label_encoder_team.fit_transform(df['team1'])
df['team2_encoded'] = label_encoder_team.fit_transform(df['team2'])
df['venue_encoded'] = label_encoder_venue.fit_transform(df['venue'])
df['winner_encoded'] = label_encoder_winner.fit_transform(df['winner'])

# Define features and target
X = df[['team1_encoded', 'team2_encoded', 'venue_encoded']]
y = df['winner_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training and prediction function
def train_predict(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write(f"### {model_name} Accuracy: {accuracy:.2f}")
    
    # Plot confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    st.pyplot(fig)
    
    return accuracy

# User input section
st.write("Enter the match details to predict the winner:")
team1 = st.selectbox("Select Team 1:", df['team1'].unique())
team2 = st.selectbox("Select Team 2:", df['team2'].unique())
venue = st.selectbox("Select Venue:", df['venue'].unique())

# Helper function to safely encode labels, handling unseen labels
def safe_label_encode(label, encoder, default_value=-1):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    else:
        st.write(f"Warning: '{label}' is not in the training data. Assigning default value.")
        return default_value

# Encode user input, handling unseen labels
team1_encoded = safe_label_encode(team1, label_encoder_team)
team2_encoded = safe_label_encode(team2, label_encoder_team)
venue_encoded = safe_label_encode(venue, label_encoder_venue)

# Create input feature for prediction
input_data = np.array([[team1_encoded, team2_encoded, venue_encoded]])

# Initialize models
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
svm = SVC()

# Model evaluation
st.write("## Model Results")

# Train and evaluate Decision Tree
train_predict(decision_tree, "Decision Tree")

# Train and evaluate Random Forest
train_predict(random_forest, "Random Forest")

# Train and evaluate Support Vector Machine
train_predict(svm, "Support Vector Machine")

# Make prediction with the Random Forest (or choose the best performing model)
best_model = random_forest
prediction = best_model.predict(input_data)
predicted_winner = label_encoder_winner.inverse_transform(prediction)[0]

# Display prediction
st.write(f"## Predicted Winner: {predicted_winner}")
