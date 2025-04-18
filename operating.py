import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def generate_data(size=2000):
    np.random.seed(42)

    cpu_burst = np.random.randint(1, 21, size)
    arrival = np.random.randint(0, 51, size)
    priority = np.random.randint(1, 11, size)
    io_bound = np.random.randint(0, 2, size)
    preemptive = np.random.randint(0, 2, size)
    cpu_load = np.random.uniform(0, 100, size)
    memory_usage = np.random.uniform(0, 100, size)

    def decide_scheduling(burst, prio, io, preempt):
        if io:
            return "Round Robin"
        if prio <= 3:
            return "Preemptive Priority Scheduling" if preempt else "Non-Preemptive Priority Scheduling"
        if burst <= 5:
            return "SJF"
        return "FCFS"

    scheduling = [
        decide_scheduling(b, p, i, pr) 
        for b, p, i, pr in zip(cpu_burst, priority, io_bound, preemptive)
    ]

    df = pd.DataFrame({
        "CPU_Burst": cpu_burst,
        "Arrival_Time": arrival,
        "Priority": priority,
        "IO_Bound": io_bound,
        "Preemptive": preemptive,
        "CPU_Load": cpu_load,
        "Memory_Usage": memory_usage,
        "Scheduling_Type": scheduling
    })

    return df

def train_classifier(df):
    encoder = LabelEncoder()
    df['Target'] = encoder.fit_transform(df['Scheduling_Type'])

    features = df.drop(columns=['Scheduling_Type', 'Target'])
    labels = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    return clf, encoder, X_test, y_test

def predict(clf, encoder, inputs):
    df_input = pd.DataFrame([inputs], columns=[
        "CPU_Burst", "Arrival_Time", "Priority", "IO_Bound", 
        "Preemptive", "CPU_Load", "Memory_Usage"
    ])
    label = clf.predict(df_input)[0]
    return encoder.inverse_transform([label])[0]

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

def show_conf_matrix(y_true, y_pred, encoder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def show_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

def main():
    print("Generating dataset...")
    df = generate_data()

    print("Training model...")
    model, encoder, X_test, y_test = train_classifier(df)

    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=encoder.classes_))
    
    show_conf_matrix(y_test, preds, encoder)
    show_feature_importance(model, X_test.columns)

    print("\nStarting real-time prediction...")

    true_list = []
    pred_list = []

    while True:
        idx = np.random.randint(0, len(df))
        row = df.iloc[idx]

        cpu_now = get_cpu_usage()
        mem_now = get_memory_usage()

        input_features = [
            row["CPU_Burst"], row["Arrival_Time"], row["Priority"],
            row["IO_Bound"], row["Preemptive"], cpu_now, mem_now
        ]

        actual = row["Scheduling_Type"]
        pred = predict(model, encoder, input_features)

        true_list.append(actual)
        pred_list.append(pred)

        pid = np.random.randint(1000, 9999)
        print(f"Process {pid}: Predicted - {pred} | Actual - {actual}")

        if len(pred_list) % 100 == 0:
            show_conf_matrix(true_list, pred_list, encoder)

        cont = input("Continue? (y/n): ").strip().lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()