import pandas as pd  # for data analysis
from ydata_profiling import ProfileReport  # for data profiling and visualization
import os  # for file management
from sklearn.impute import KNNImputer  # for imputing missing values for numerical
from sklearn.impute import SimpleImputer  # for imputing missing values for non-numerical
import matplotlib.pyplot as plt  # for data visualization
import seaborn as sns  # for data visualization and statistical analysis

# normalization
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.preprocessing import OneHotEncoder  # for one-hot encoding
from sklearn.preprocessing import LabelEncoder  # for label encoding
from sklearn.model_selection import train_test_split  # for train and test split

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
path = r'C:\Users\admin\OneDrive\문서\AIML\AIML PROJECT-3\synthetic_food_dataset_imbalanced.csv'

try:
    food_data = pd.read_csv(path)
except Exception as e:
    print("Error loading file:", e)

# checking null values and duplicates

# print('null_values:', food_data.isnull().sum().sum())
# print('duplicate_value:', food_data.duplicated().sum())

'''
till now i find the null values and duplicates in the dataset but i want to remove the duplicates
and fill the null value. I can use pandas fillna() method to do that but i want more accurate results
for model training so i can use KNNImputer() method to impute the missing values.

to achieve this i need to do the following steps:
1. separate numeric and non-numeric columns (because KNNImputer only works on numeric columns)
2. apply KNNImputer() method on numeric columns
'''

# Drop duplicates
food_data = food_data.drop_duplicates()

# Separate numeric and non-numeric columns
numeric_col = food_data.select_dtypes(include=['int64', 'float64']).columns
non_numeric_col = food_data.select_dtypes(exclude=['int64', 'float64']).columns

# Apply KNNImputer only on numeric columns
imputer = KNNImputer(n_neighbors=5)
food_data_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(food_data[numeric_col]),
    columns=numeric_col
)

'''
till now the numerical columns are imputed but the non-numeric columns are not imputed.
i need to fill the missing values in non-numeric columns. The dtypes are boolean, object and category.
so i use SimpleImputer to fill the missing values in non-numeric columns.
'''

# Fill missing for non-numeric columns
imputer_cat = SimpleImputer(strategy='most_frequent')
food_data_category_imputed = pd.DataFrame(
    imputer_cat.fit_transform(food_data[['Meal_Type', 'Preparation_Method', 'Is_Vegan', 'Is_Gluten_Free']]),
    columns=['Meal_Type', 'Preparation_Method', 'Is_Vegan', 'Is_Gluten_Free']
)

# Fill Food_Name (text column)
food_data_text_imputed = pd.DataFrame(
    food_data['Food_Name'].fillna("Unknown"),
    columns=['Food_Name']
)

# Combine all imputed data
food_all_data_imputed = pd.concat([
    food_data_numeric_imputed.reset_index(drop=True),
    food_data_category_imputed.reset_index(drop=True),
    food_data_text_imputed.reset_index(drop=True)
], axis=1)

# Drop duplicates again
food_all_data_imputed = food_all_data_imputed.drop_duplicates()

# Ensure boolean flags are numeric
food_all_data_imputed['Is_Vegan'] = food_all_data_imputed['Is_Vegan'].astype(int)
food_all_data_imputed['Is_Gluten_Free'] = food_all_data_imputed['Is_Gluten_Free'].astype(int)

# Save cleaned dataset
clean_path = r'C:\Users\admin\OneDrive\문서\AIML\AIML PROJECT-3\synthetic_food_dataset_clean.csv'
food_all_data_imputed.to_csv(clean_path, index=False)

'''
till now i see outliers on ydata_profilling but i want to visualize
the outliers so i use boxplot to visualize the outliers.
'''

# Boxplot for outliers (optional)
# for col in numeric_col:
#     plt.figure(figsize=(6, 3))
#     sns.boxplot(x=food_all_data_imputed[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

'''
now i found outliers in the dataset but if chicken was deep fried
means it has high fat content so calories also high it may go 1200-1400

so, i decided that to keep the outliers as same as the original dataset
'''

'''
now i am going to train the model but before that i need to do the following steps:
1. feature scaling (normalization)
2. one hot encoding
'''

# Scaling numerical columns
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(food_all_data_imputed[numeric_col])
scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_col)

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_cat = encoder.fit_transform(food_all_data_imputed[['Meal_Type', 'Preparation_Method']])
encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(['Meal_Type', 'Preparation_Method'])
)

# Keep boolean columns
bool_cols = ['Is_Vegan', 'Is_Gluten_Free']
bool_df = food_all_data_imputed[bool_cols].reset_index(drop=True)

'''
till now i completed preprocessing but now i am going to train the model
1. split data
2. train model
3. evaluate model
'''

# Prepare final X and y
X = pd.concat([
    scaled_numeric_df.reset_index(drop=True),
    encoded_cat_df.reset_index(drop=True),
    bool_df.reset_index(drop=True)
], axis=1)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(food_all_data_imputed['Food_Name'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss'), 
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train & evaluate models
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append([name, acc, prec, rec, f1])

    print(f"Accuracy: {acc*100:.4f}%")
    print(f"Precision: {prec*100:.4f}%")
    print(f"Recall: {rec*100:.4f}%")
    print(f"F1-score: {f1*100:.4f}%")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# Final results table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
print("\nModel Comparison Table:")
print(results_df)

# test with a new data
any_model = models["SVM"]    

def predict_food_item(
    Calories, Protein, Fat, Carbs, Sugar, Fiber,
    Cholesterol, Sodium, Glycemic_Index, Water_Content, Serving_Size,
    Meal_Type, Preparation_Method, Is_Vegan, Is_Gluten_Free
):

    #Build input dataframe
    new_data = pd.DataFrame([[
        Calories, Protein, Fat, Carbs, Sugar, Fiber,
        Cholesterol, Sodium, Glycemic_Index, Water_Content, Serving_Size,
        Meal_Type, Preparation_Method, Is_Vegan, Is_Gluten_Free
    ]], 
    columns=[
        "Calories","Protein","Fat","Carbs","Sugar","Fiber",
        "Cholesterol","Sodium","Glycemic_Index","Water_Content","Serving_Size",
        "Meal_Type","Preparation_Method","Is_Vegan","Is_Gluten_Free"
    ])

    # Scale numeric columns
    new_scaled_nums = scaler.transform(new_data[numeric_col])

    # Convert to DF
    new_scaled_nums_df = pd.DataFrame(new_scaled_nums, columns=numeric_col)

    # One-hot encode categorical columns
    new_cat_encoded = encoder.transform(new_data[["Meal_Type", "Preparation_Method"]])
    new_cat_df = pd.DataFrame(
        new_cat_encoded,
        columns=encoder.get_feature_names_out(["Meal_Type", "Preparation_Method"])
    )

    #Keep boolean columns
    new_bool_df = new_data[["Is_Vegan", "Is_Gluten_Free"]].reset_index(drop=True)

    #Merge everything
    final_input = pd.concat([new_scaled_nums_df, new_cat_df, new_bool_df], axis=1)

    #Predict
    pred = any_model.predict(final_input)[0]

    # Convert encoded label → original label
    predicted_name = label_encoder.inverse_transform([pred])[0]

    return predicted_name


result = predict_food_item(
    Calories=450,
    Protein=30,
    Fat=15,
    Carbs=60,
    Sugar=12,
    Fiber=5,
    Cholesterol=80,
    Sodium=700,
    Glycemic_Index=85,
    Water_Content=120,
    Serving_Size=250,
    Meal_Type="lunch",
    Preparation_Method="fried",
    Is_Vegan=0,
    Is_Gluten_Free=1
)

print("Predicted Food Name:", result)
