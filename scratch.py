import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset from Excel file
df = pd.read_excel("C:\\Users\\PrinceАмирBubblegum\\Desktop\\Data Capstone (2).xlsx")

# Convert Date of Admission to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], format='%d.%m.%Y')

# Encode categorical variables
label_encoders = {}
for column in ["Gender", "Blood Type", "Medication", "Test Results", "Admission Type"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Feature Engineering
X = df.drop(["Name", "Medical Condition"], axis=1)
X["Year"] = X["Date of Admission"].dt.year
X["Month"] = X["Date of Admission"].dt.month
X["Day"] = X["Date of Admission"].dt.day
X.drop("Date of Admission", axis=1, inplace=True)

# Split dataset into features and target variable
y = df["Medical Condition"]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Example usage of the trained model for prediction
# Let's say we have a new sample called 'new_sample' for which we want to predict the medical condition
# You need to preprocess 'new_sample' in the same way as the training data and then use the trained model for prediction
# For example:
new_sample = pd.DataFrame({
    "Age": [40],
    "Gender": ["Male"],
    "Blood Type": ["A+"],
    "Admission Type": ["Emergency"],
    "Medication": ["Aspirin"],
    "Test Results": ["Normal"],
    "Date of Admission": ["10.01.2024"]
})
new_sample["Date of Admission"] = pd.to_datetime(new_sample["Date of Admission"], format='%d.%m.%Y')
new_sample["Gender"] = label_encoders["Gender"].transform(new_sample["Gender"])
new_sample["Blood Type"] = label_encoders["Blood Type"].transform(new_sample["Blood Type"])
new_sample["Admission Type"] = label_encoders["Admission Type"].transform(new_sample["Admission Type"])
new_sample["Medication"] = label_encoders["Medication"].transform(new_sample["Medication"])
new_sample["Test Results"] = label_encoders["Test Results"].transform(new_sample["Test Results"])
new_sample["Year"] = new_sample["Date of Admission"].dt.year
new_sample["Month"] = new_sample["Date of Admission"].dt.month
new_sample["Day"] = new_sample["Date of Admission"].dt.day
new_sample.drop("Date of Admission", axis=1, inplace=True)
predicted_condition = model.predict(new_sample)
print("Predicted Medical Condition:", predicted_condition)



df = pd.read_excel("C:\\Users\\PrinceАмирBubblegum\\Desktop\\Data Capstone (2).xlsx")

# Convert Date of Admission to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], format='%d.%m.%Y')

# Encode categorical variables
label_encoders = {}
for column in ["Gender", "Blood Type", "Medication", "Test Results", "Admission Type"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Feature Engineering
X = df.drop(["Name", "Medical Condition"], axis=1)
X["Year"] = X["Date of Admission"].dt.year
X["Month"] = X["Date of Admission"].dt.month
X["Day"] = X["Date of Admission"].dt.day
X.drop("Date of Admission", axis=1, inplace=True)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target variable
y = df["Medical Condition"]
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
input_shape = X_train.shape[1]
# Build deep learning model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Convert predictions back to original labels
y_pred_original = label_encoder_y.inverse_transform(y_pred.ravel())

# Convert test labels back to original labels
y_test_original = label_encoder_y.inverse_transform(y_test)

print("\nClassification Report:")
print(classification_report(y_test_original, y_pred_original))