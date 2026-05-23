from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
    "Temperature": [30, 32, 35, 28, 25, 40, 38],
    "Humidity": [70, 65, 60, 80, 85, 50, 55],
    "Rain": [1, 0, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[["Temperature", "Humidity"]]
y = df["Rain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[29, 75]])
print("Rain Prediction:", prediction)