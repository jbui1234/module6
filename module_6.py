import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("mental_health.csv")  

data = df.copy()

data['Gender'] = data['Gender'].str.lower()
data['Gender'] = data['Gender'].replace({
    'male': 'male', 'm': 'male', 'man': 'male', 'cis male': 'male',
    'female': 'female', 'f': 'female', 'woman': 'female', 'cis female': 'female'
})
data['Gender'] = data['Gender'].apply(lambda x: x if x in ['male', 'female'] else 'other')

data = data.dropna(subset=['treatment'])

data['work_interfere'] = data['work_interfere'].fillna("Don't know")
data['self_employed'] = data['self_employed'].fillna("Don't know")

features = [
    'Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'anonymity',
    'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor'
]
target = 'treatment'

data = data[features + [target]]

data_encoded = pd.get_dummies(data, columns=[col for col in features if data[col].dtype == 'object'])

label_encoder = LabelEncoder()
data_encoded[target] = label_encoder.fit_transform(data_encoded[target])

X = data_encoded.drop(columns=[target])
y = data_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

evaluation_metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1
}

evaluation_metrics, conf_matrix, importances.head(10)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

X_test_copy = X_test.copy()
X_test_copy["Actual"] = y_test
X_test_copy["Predicted"] = y_pred
X_test_copy["Correct"] = X_test_copy["Actual"] == X_test_copy["Predicted"]
misclassified_samples = X_test_copy[X_test_copy["Correct"] == False].sample(5, random_state=42)

class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

full_importances = importances.reset_index()
full_importances.columns = ['Feature', 'Importance']

print("\nEvaluation Metrics")
for metric, value in evaluation_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nConfusion Matrix")
print(conf_matrix)

print("\n 5 Misclassified Samples ")
print(misclassified_samples.head())

misclassified_samples, class_report_df.head(), full_importances.head(20)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='work_interfere', hue='treatment', order=['Never', 'Rarely', 'Sometimes', 'Often', "Don't know"])
plt.title('Treatment vs. Self-Reported Work Interference')
plt.xlabel('Work Interference')
plt.ylabel('Count')
plt.legend(title='Sought Treatment')
plt.tight_layout()
plt.show()

df_model = df[['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
               'no_employees', 'remote_work', 'tech_company', 'benefits',
               'care_options', 'wellness_program', 'seek_help', 'anonymity',
               'leave', 'mental_health_consequence', 'phys_health_consequence',
               'coworkers', 'supervisor', 'treatment']]
df_model_encoded = pd.get_dummies(df_model, drop_first=True)
correlation = df_model_encoded.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation[['treatment_Yes']].sort_values(by='treatment_Yes', ascending=False), annot=True, cmap='coolwarm')
plt.title('Correlation of Features with Mental Health Treatment')
plt.tight_layout()
plt.show()