import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.io import arff
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import scikitplot as skplt
import xgboost as xgb

def preprocess_dataset(file_path):
    """
    Description:
    ------------
        Preprocesses the input dataset by encoding
        categorical variables, scaling numerical features,
        and splitting the data into training and testing sets.

    Parameters:
    ------------
        I   file_path (str): Path to the ARFF file containing the dataset.

    Returns:
    ------------
        I   X_train (np.ndarray): Training features.
        II  X_test (np.ndarray): Testing features.
        III y_train (np.ndarray): Training labels.
        IV  y_test (np.ndarray): Testing labels.
    """
    # Loading and converting the ARFF data to a pandas DataFrame
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings in categorical columns
    categorical_cols = ["seismic", "seismoacoustic", "shift", "ghazard", "class"]
    for col in categorical_cols:
        df[col] = df[col].str.decode("utf-8")

    # Encode the target variable ('class') into integers
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])

    # One-hot encode other categorical variables
    df = pd.get_dummies(df, columns=["seismic", "seismoacoustic", "shift", "ghazard"], drop_first=True)

    # Separate features and target
    X = df.drop("class", axis=1)
    y = df["class"]

    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

    return X_train, X_test, y_train, y_test, X.columns

def decision_tree(X_train, X_test, y_train, y_test, feature_names, max_depth):
    """
    Description:
    ------------
        Trains a Decision Tree classifier, evaluates its performance on a test set,
        and visualizes the tree structure.

    Parameters:
    ------------
        I   X_train, X_test (np.ndarray): Training and testing features.
        II  y_train, y_test (np.ndarray): Training and testing labels.
        III feature_names (Index): Names of the feature columns for visualization.
        IV  max_depth (int): Maximum depth of the decision tree.
    """
    # Initialize and train the decision tree classifier
    clf = DecisionTreeClassifier(class_weight="balanced", max_depth=max_depth, random_state=123)
    clf.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = clf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Validation Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)


    plt.figure(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, class_names=["Class 0", "Class 1"], filled=True, fontsize=12)

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "Decision_tree.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved to {save_path}")


def xgboost(X_train, X_test, y_train, y_test, feature_names):
    """
    Description:
    ------------
        Trains an XGBoost classifier, evaluates its performance on a test set,
        and prints evaluation metrics.

    Parameters:
    ------------
        I   X_train, X_test (np.ndarray): Training and testing features.
        II  y_train, y_test (np.ndarray): Training and testing labels.
        III feature_names (Index): Names of the feature columns.
        """
    scale_pos_weight = 1943 / 124
    #xg_clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=123)
    xg_clf = xgb.XGBClassifier(random_state=123)

    xg_clf.fit(X_train,y_train)
    #y_test = xg_clf.predict(X_test)

    y_pred = xg_clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Validation Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)




#Preprocess dataset
file_path = "./dataset/dataset"
X_train, X_test, y_train, y_test, feature_names = preprocess_dataset(file_path)

decision_tree(X_train, X_test, y_train, y_test, feature_names, 4)
xgboost(X_train, X_test, y_train, y_test, feature_names)
