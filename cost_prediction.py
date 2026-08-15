import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_excel("medical_insurance_dataset.xlsx")
# print(df)

# -----------------------------------Handling Missing Value--------------------------------------------

# df.info()

# print(df.isnull().sum())
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age"] = df["age"].fillna(df["age"].mean())

# print(df["bmi"].dtype)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# plt.boxplot(df["bmi"])
# plt.show()

df["smoker"] = df["smoker"].fillna(df["smoker"].mode()[0])

# print(df.isnull().sum())

# print(df.duplicated().sum())
# print(df.drop_duplicates())

# ----------------------------------------Handing Case--------------------------------------------------
df["sex"] = df["sex"].str.strip().str.title()
# print(df["sex"].unique())

df["region"] = df["region"].str.strip()
# print(df["region"].unique())

# -----------------------------------------Encoding-----------------------------------------------------

le = LabelEncoder()

df["smoker"] = le.fit_transform(df["smoker"].astype(str))
df["sex"] = le.fit_transform(df["sex"].astype(str))
# print(le.classes_)
# print(df.head())

ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df[["region"]])
df[ohe.get_feature_names_out()] = encoded
df.drop("region", axis=1, inplace=True)
# print(df.head())

# ----------------------------------------Handle outlier-----------------------------------------------


def handle_outlier(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return series.clip(lower, upper)


# print(df["bmi"].describe())
df["bmi"] = handle_outlier(df["bmi"])
# print(df["bmi"].describe())

# print(df["age"].describe())
df["age"] = handle_outlier(df["age"])
# print(df["age"].describe())

# ---------------------------------------Split dataset ----------------------------------------------

x = df.drop("charges", axis=1)
y = df["charges"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -------------------------------------------Scaling--------------------------------------------------

scale = StandardScaler()

x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# --------------------------------------------Training-------------------------------------------------

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# print(y_pred[:5])

# --------------------------------------------evaluation------------------------------------------------

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"MSE : {mse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R2_Score : {r2:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"Accuracy : {r2 * 100:.2f} %")

# ---------------------------------------prediction on new data-------------------------------------------


new_data = pd.DataFrame(
    {
        "age": [35],
        "sex": [1],
        "bmi": [27.5],
        "children": [2],
        "smoker": [0],
        "region_northeast": [1],
        "region_northwest": [0],
        "region_southeast": [0],
        "region_southwest": [0],
    }
)

new_data = scale.transform(new_data)
prediction = model.predict(new_data)

# print("Predicted Charges on new data :", prediction[0])

# --------------------------------------------------plot graph------------------------------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual charges")
plt.ylabel("Predicated charges")
plt.title("Actual vs Predicated")
plt.savefig("screenshots/Actual vs Predicated", dpi=300)
plt.show()


plt.bar(x.columns, model.coef_)
plt.xticks(rotation=45)
plt.savefig("screenshots/model co-efficient", dpi=300)
plt.show()

# -------------------------------------save the model----------------------------------------------
joblib.dump(model, "insurance_model.pkl")
