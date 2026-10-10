# %%
import pandas as pd

data = pd.read_csv("games_dataset(2024).csv")

data.head()

# %%
data.shape

# %%
data.columns

# %%
data.info()

# %%
data.isnull().sum()

# %%
data.describe()

# %%
data.duplicated().sum()

# %%
data["title"].duplicated().sum()

# %%
print("Unique games:", data["title"].nunique())
print("Unique platforms:", data["console"].nunique())
print("Unique genres:", data["genre"].nunique())
print("Unique publishers:", data["publisher"].nunique())
print("Unique developers:", data["developer"].nunique())

# %%
missing = data.isnull().sum()

missing_percentage = (missing / len(data)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percentage
})

missing_table.sort_values("Percentage", ascending=False)

# %%
data["total_sales"].describe()

# %%
data["total_sales"].isnull().sum()

# %%
print(data["genre"].value_counts())

# %%
print(data["console"].value_counts().head(20))

# %%
print(data["publisher"].value_counts().head(20))

# %%
data.drop(columns=["img"], inplace=True)

# %%
data.columns

# %%
model_data = data.dropna(subset=["total_sales"]).copy()

# %%
print("Original rows:", len(data))
print("Rows for modeling:", len(model_data))

# %%
model_data.isnull().sum()

# %%
missing_model = model_data.isnull().sum()

missing_model = missing_model[missing_model > 0].sort_values(ascending=False)

missing_model

# %%
model_data[["genre", "publisher", "developer", "critic_score", "release_date"]].isnull().sum()

# %%
model_data[["genre", "publisher", "developer", "critic_score", "release_date"]].isnull().sum()

# %%
categorical_columns = ["genre", "publisher", "developer"]

for column in categorical_columns:
    model_data[column] = model_data[column].fillna("Unknown")

# %%
model_data[categorical_columns].isnull().sum()

# %%
median_score = model_data["critic_score"].median()

model_data["critic_score"] = model_data["critic_score"].fillna(median_score)

# %%
model_data["critic_score"].isnull().sum()

# %%
model_data["release_date"] = pd.to_datetime(
    model_data["release_date"],
    errors="coerce"
)

# %%
model_data["release_year"] = model_data["release_date"].dt.year

# %%
model_data["release_month"] = model_data["release_date"].dt.month

# %%
model_data[["release_date", "release_year", "release_month"]].head()

# %%
model_data.info()

# %%
model_data.isnull().sum()

# %%
model_data["total_sales"].describe()

# %%
model_data["total_sales"].median()

# %%
model_data["total_sales"].mean()

# %%
top_games = model_data.sort_values(
    "total_sales",
    ascending=False
).head(10)

top_games[["title", "console", "genre", "total_sales"]]

# %%
import matplotlib.pyplot as plt

# %%
top_10 = model_data.sort_values(
    "total_sales",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10["title"],
    top_10["total_sales"]
)

plt.xlabel("Total Sales (Millions)")
plt.ylabel("Game")
plt.title("Top 10 Best-Selling Video Games")

plt.gca().invert_yaxis()

plt.show()

# %%
genre_sales = model_data.groupby("genre")["total_sales"].sum()

genre_sales = genre_sales.sort_values(ascending=False)

genre_sales

# %%
plt.figure(figsize=(10, 6))

plt.bar(
    genre_sales.index,
    genre_sales.values
)

plt.xlabel("Genre")
plt.ylabel("Total Sales (Millions)")
plt.title("Total Video Game Sales by Genre")

plt.xticks(rotation=45)

plt.show()

# %%
avg_genre_sales = model_data.groupby("genre")["total_sales"].mean()

avg_genre_sales = avg_genre_sales.sort_values(ascending=False)

avg_genre_sales

# %%
plt.figure(figsize=(10, 6))

plt.bar(
    avg_genre_sales.index,
    avg_genre_sales.values
)

plt.xlabel("Genre")
plt.ylabel("Average Sales per Game (Millions)")
plt.title("Average Game Sales by Genre")

plt.xticks(rotation=45)

plt.show()

# %%
platform_sales = model_data.groupby("console")["total_sales"].sum()

platform_sales = platform_sales.sort_values(ascending=False).head(15)

platform_sales

# %%
plt.figure(figsize=(10, 6))

plt.bar(
    platform_sales.index,
    platform_sales.values
)

plt.xlabel("Platform")
plt.ylabel("Total Sales (Millions)")
plt.title("Top 15 Platforms by Total Game Sales")

plt.xticks(rotation=45)

plt.show()

# %%
plt.figure(figsize=(8, 6))

plt.scatter(
    model_data["critic_score"],
    model_data["total_sales"],
    alpha=0.5
)

plt.xlabel("Critic Score")
plt.ylabel("Total Sales (Millions)")
plt.title("Critic Score vs Game Popularity")

plt.show()

# %%
correlation = model_data["critic_score"].corr(
    model_data["total_sales"]
)

print("Correlation:", correlation)

# %%
games_per_year = model_data.groupby("release_year")["title"].count()

games_per_year.tail(20)

# %%
plt.figure(figsize=(12, 6))

plt.plot(
    games_per_year.index,
    games_per_year.values
)

plt.xlabel("Release Year")
plt.ylabel("Number of Games")
plt.title("Number of Games Released per Year")

plt.show()

# %%
sales_per_year = model_data.groupby("release_year")["total_sales"].sum()

sales_per_year.tail(20)

# %%
plt.figure(figsize=(12, 6))

plt.plot(
    sales_per_year.index,
    sales_per_year.values
)

plt.xlabel("Release Year")
plt.ylabel("Total Sales (Millions)")
plt.title("Total Video Game Sales by Year")

plt.show()

# %%
regional_sales = model_data[
    ["na_sales", "jp_sales", "pal_sales", "other_sales"]
].sum()

regional_sales

# %%
plt.figure(figsize=(8, 6))

plt.bar(
    regional_sales.index,
    regional_sales.values
)

plt.xlabel("Region")
plt.ylabel("Sales (Millions)")
plt.title("Video Game Sales by Region")

plt.show()

# %%
publisher_sales = (
    model_data
    .groupby("publisher")["total_sales"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

publisher_sales

# %%
plt.figure(figsize=(10, 6))

plt.bar(
    publisher_sales.index,
    publisher_sales.values
)

plt.xlabel("Publisher")
plt.ylabel("Total Sales (Millions)")
plt.title("Top 15 Publishers by Total Game Sales")

plt.xticks(rotation=75)

plt.show()

# %%
model_data.columns

# %%
ml_data = model_data.copy()

# %%
ml_data = ml_data.drop(
    columns=[
        "total_sales",
        "na_sales",
        "jp_sales",
        "pal_sales",
        "other_sales",
        "title",
        "release_date"
    ]
)

# %%
ml_data.head()

# %%
ml_data.head()

# %%
X = ml_data

# %%
y = model_data["total_sales"]

# %%
print("X shape:", X.shape)
print("y shape:", y.shape)

# %%
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# %%
from sklearn.linear_model import LinearRegression

# %%
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# %%
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# %%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ],
    remainder="drop"
)

# %%
from sklearn.linear_model import LinearRegression

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# %%
linear_model.fit(X_train, y_train)

# %%
y_pred_linear = linear_model.predict(X_test)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = mean_squared_error(y_test, y_pred_linear) ** 0.5
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression Results")
print("MAE:", mae_linear)
print("RMSE:", rmse_linear)
print("R²:", r2_linear)

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# %%
random_forest_model.fit(X_train, y_train)

# %%
y_pred_rf = random_forest_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression Results")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R²:", r2_rf)

# %%
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae_linear, mae_rf],
    "RMSE": [rmse_linear, rmse_rf],
    "R2 Score": [r2_linear, r2_rf]
})

comparison

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.4, label="Linear Regression")
plt.scatter(y_test, y_pred_rf, alpha=0.4, label="Random Forest")

max_value = max(y_test.max(), y_pred_linear.max(), y_pred_rf.max())
plt.plot([0, max_value], [0, max_value], "r--", label="Perfect Prediction")

plt.xlabel("Actual Total Sales (Millions)")
plt.ylabel("Predicted Total Sales (Millions)")
plt.title("Actual vs Predicted Game Sales")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Get the trained Random Forest model
rf = random_forest_model.named_steps["model"]

# Get the preprocessed feature names
feature_names = (
    random_forest_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Create a feature importance DataFrame
importance_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

# Sort by importance
importance_data = importance_data.sort_values(
    by="Importance",
    ascending=False
)

importance_data.head(15)

# %%
top_features = importance_data.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features for Predicting Game Popularity")
plt.tight_layout()
plt.show()

# %%
comparison

# %%
best_model_name = comparison.sort_values(
    by="RMSE",
    ascending=True
).iloc[0]["Model"]

print("Best Model based on RMSE:", best_model_name)
print("\nModel Comparison:")
print(comparison.to_string(index=False))

# %%
model_data.to_csv("cleaned_game_dataset.csv", index=False)

print("Cleaned dataset saved successfully!")

# %%
print("========== GAME POPULARITY ANALYSIS ==========")

print("\n1. Dataset Overview")
print("Number of games analyzed:", len(model_data))

print("\n2. Average Total Sales (Millions)")
print(model_data["total_sales"].mean())

print("\n3. Median Total Sales (Millions)")
print(model_data["total_sales"].median())

print("\n4. Top 10 Best-Selling Games")
print(
    model_data.nlargest(10, "total_sales")[
        ["title", "console", "genre", "total_sales"]
    ].to_string(index=False)
)

print("\n5. Model Comparison")
print(comparison.to_string(index=False))

print("\n6. Top 10 Predictive Features")
print(importance_data.head(10).to_string(index=False))

# %%
with open("game_popularity_summary.txt", "w", encoding="utf-8") as file:
    file.write("GAME POPULARITY DATA SCIENCE PROJECT\n\n")

    file.write("MODEL COMPARISON\n")
    file.write(comparison.to_string(index=False))

    file.write("\n\nTOP 10 PREDICTIVE FEATURES\n")
    file.write(importance_data.head(10).to_string(index=False))

print("Project summary saved successfully!")


