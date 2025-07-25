import pandas as pd #reading data
from sklearn.model_selection import train_test_split #split data into this model
from sklearn.preprocessing import LabelEncoder, StandardScaler #convert categorical value into num
from sklearn.svm import SVC #svm algo
from sklearn.pipeline import make_pipeline #combine processing & model into step
from sklearn.metrics import accuracy_score, classification_report
#evalute its performance
import joblib 

# for Loading dataset
df = pd.read_csv("handbags_two_brands_cleaned00.csv")
df.columns = df.columns.str.strip()  #if there are some extra space it Remove spaces in column names

# converting handbag type (subcategory) from text to number
le_sub = LabelEncoder()
df['Subcategory'] = le_sub.fit_transform(df['Subcategory'])

# converting brand names to numbers (LuxeCraft = 0, StyleNest = 1)
le_brand = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])  # 0 = LuxeCraft, 1 = StyleNest

# selecting the features i’ll use for prediction (X) and the brand as output (y)
X = df[['Subcategory', 'Price', 'Rating']]
y = df['Brand']

# splitting the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# creating the model pipeline: first it will scale the features, then apply SVM
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

#Train the model
model.fit(X_train, y_train)
# making predictions using the test data
y_pred = model.predict(X_test)

# checking the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
brand_names = le_brand.inverse_transform([0, 1])#convert the brand name to num bcz it is easy for making report
report = classification_report(y_test, y_pred, target_names=brand_names)
print("=== Handbag Brand Prediction ===")
print("Accuracy:", round(accuracy * 100, 2), "%")#for percentage
print()#blank line
print(report)

# saving everything for Flask app: model + label encoders
joblib.dump(model, "svm_brand_model.pkl")
joblib.dump(le_sub, "subcategory_encoder.pkl")
joblib.dump(le_brand, "brand_label_encoder.pkl")
