import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def generate_dataset(data_size=2000):
    np.random.seed(42)
    cpu_burst = np.random.randint(1, 21, data_size)
    arrival_time = np.random.randint(0, 51, data_size)
    priority = np.random.randint(1, 11, data_size)
    io_bound = np.random.randint(0, 2, data_size)
    preemptive = np.random.randint(0, 2, data_size)
    cpu_load = np.random.uniform(0, 100, data_size)  
    memory_usage = np.random.uniform(0, 100, data_size)  
    
    def assign_scheduling(burst, priority, io, preemptive):
        if io == 1:
            return "Round Robin"
        elif priority <= 3 and preemptive == 1:
            return "Preemptive Priority Scheduling"
        elif priority <= 3:
            return "Non-Preemptive Priority Scheduling"
        elif burst <= 5:
            return "SJF"
        else:
            return "FCFS"
    
    scheduling_type = [assign_scheduling(b, p, i, pre) for b, p, i, pre in zip(cpu_burst, priority, io_bound, preemptive)]
    
    df = pd.DataFrame({
        "CPU_Burst": cpu_burst,
        "Arrival_Time": arrival_time,
        "Priority": priority,
        "IO_Bound": io_bound,
        "Preemptive": preemptive,
        "CPU_Load": cpu_load,
        "Memory_Usage": memory_usage,
        "Scheduling_Type": scheduling_type
    })
    return df

def train_model(df):
    label_encoder = LabelEncoder()
    df["Scheduling_Type_Label"] = label_encoder.fit_transform(df["Scheduling_Type"])
    
    X = df[["CPU_Burst", "Arrival_Time", "Priority", "IO_Bound", "Preemptive", "CPU_Load", "Memory_Usage"]] 
    y = df["Scheduling_Type_Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    return model, label_encoder, X_test, y_test

def predict_scheduling(model, label_encoder, cpu_burst, arrival_time, priority, io_bound, preemptive, cpu_load, memory_usage):
    input_data = np.array([[cpu_burst, arrival_time, priority, io_bound, preemptive, cpu_load, memory_usage]])
    input_df = pd.DataFrame(input_data, columns=["CPU_Burst", "Arrival_Time", "Priority", "IO_Bound", "Preemptive", "CPU_Load", "Memory_Usage"])
    predicted_label = model.predict(input_df)[0]
    return label_encoder.inverse_transform([predicted_label])[0]

def plot_confusion_matrix(conf_matrix, label_encoder):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    plt.figure(figsize=(8, 5))
    
    sns.barplot(x=feature_importance, y=feature_names, palette="viridis")
    
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Random Forest Model")
    plt.show()

def monitor_cpu():
    cpu_percent = psutil.cpu_percent(interval=1)  
    cpu_load_per_core = psutil.cpu_percent(interval=1, percpu=True)  
    return cpu_percent, cpu_load_per_core

def monitor_memory():
    memory = psutil.virtual_memory()
    memory_percent = memory.percent  
    return memory_percent

def update_confusion_matrix(true_labels, predicted_labels, label_encoder):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(conf_matrix, label_encoder)

def main():
    df = generate_dataset()
    model, label_encoder, X_test, y_test = train_model(df)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    print("\n Model Accuracy:", accuracy)
    print("\n Classification Report:\n", class_report)
    plot_confusion_matrix(conf_matrix, label_encoder)
    plot_feature_importance(model, X_test.columns)
    
    true_labels = []  
    predicted_labels = []  
    
    print("Real-Time CPU Monitoring & Process Scheduling Prediction")
    
    while True:
        cpu_percent, cpu_load = monitor_cpu()
        memory_percent = monitor_memory()
        

        for _, row in df.iterrows():
            pid = np.random.randint(1000, 9999)  
            cpu_burst = row['CPU_Burst']
            arrival_time = row['Arrival_Time']
            priority = row['Priority']
            io_bound = row['IO_Bound']
            preemptive = row['Preemptive']
            

            actual_scheduling = row['Scheduling_Type']
            true_labels.append(actual_scheduling)
            

            prediction = predict_scheduling(model, label_encoder, cpu_burst, arrival_time, priority, io_bound, preemptive, cpu_percent, memory_percent)
            predicted_labels.append(prediction)
            print(f"Process {pid} - Predicted Scheduling Algorithm: {prediction}")
            

        if len(predicted_labels) % 100 == 0:
            update_confusion_matrix(true_labels, predicted_labels, label_encoder)
        
        cont = input("Do you want to continue monitoring and predicting? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
