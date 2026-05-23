from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
    "Depth": [10, 20, 15, 30, 25, 12, 18],
    "Magnitude": [4.5, 5.2, 4.8, 6.0, 5.5, 4.2, 5.0],
    "Damage": [20, 50, 30, 90, 70, 15, 40]
}

df = pd.DataFrame(data)

X = df[["Depth", "Magnitude"]]
y = df["Damage"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[22, 5.3]])
print("Predicted Damage Level:", prediction)