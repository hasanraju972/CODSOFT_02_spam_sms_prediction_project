import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load the dataset
file_path = 'spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Drop unnecessary columns and rename columns
data = data.iloc[:, :2]
data.columns = ['label', 'message']

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train and evaluate Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_nb = nb_classifier.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_report = classification_report(y_test, y_pred_nb, output_dict=True)
nb_conf_matrix = confusion_matrix(y_test, y_pred_nb)

# Train and evaluate Logistic Regression classifier
log_reg_classifier = LogisticRegression(max_iter=1000)
log_reg_classifier.fit(X_train_tfidf, y_train)
y_pred_log_reg = log_reg_classifier.predict(X_test_tfidf)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_report = classification_report(y_test, y_pred_log_reg, output_dict=True)
log_reg_conf_matrix = confusion_matrix(y_test, y_pred_log_reg)

# Train and evaluate SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Create a function to print the classification report and confusion matrix in table format
def print_report_and_matrix(clf_name, report, conf_matrix):
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = [
        ["Ham (0)", report['0']['precision'], report['0']['recall'], report['0']['f1-score'], report['0']['support']],
        ["Spam (1)", report['1']['precision'], report['1']['recall'], report['1']['f1-score'], report['1']['support']],
        ["Accuracy", "", "", report['accuracy'], ""],
        ["Macro Avg", report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'], report['macro avg']['support']],
        ["Weighted Avg", report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score'], report['weighted avg']['support']]
    ]
    print(f"{clf_name} Classification Report:\n")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\n{clf_name} Confusion Matrix:\n")
    print(tabulate(conf_matrix, headers=["Predicted Ham", "Predicted Spam"], showindex=["Actual Ham", "Actual Spam"], tablefmt="grid"))
    print("\n" + "="*60 + "\n")

# Print classification reports and confusion matrices
print_report_and_matrix("Naive Bayes", nb_report, nb_conf_matrix)
print_report_and_matrix("Logistic Regression", log_reg_report, log_reg_conf_matrix)
print_report_and_matrix("SVM", svm_report, svm_conf_matrix)

# Visualization functions
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Spam'], yticklabels=['Legitimate', 'Spam'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(nb_conf_matrix, 'Naive Bayes Confusion Matrix')
plot_confusion_matrix(log_reg_conf_matrix, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(svm_conf_matrix, 'SVM Confusion Matrix')
