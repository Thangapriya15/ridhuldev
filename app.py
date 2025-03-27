import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from scipy.stats import skew
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import Tk, Label, PhotoImage
from tkinter.filedialog import askopenfilename
import seaborn as sns


def calculate_image_statistics(image):
    """
    This function computes the mean, variance, standard deviation, median,
    and skewness of the image pixel intensities.
    """
    # Flatten the image to 1D array of pixel intensities
    pixels = image.flatten()

    # Calculate mean
    mean = np.mean(pixels)
    
    # Calculate variance
    variance = np.var(pixels)
    
    # Calculate standard deviation
    std_dev = np.std(pixels)
    
    # Calculate median
    median = np.median(pixels)
    
    # Calculate skewness
    skewness = skew(pixels)
    
    return [mean, variance, std_dev, median, skewness]


def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract Local Binary Pattern (LBP) features from a grayscale image.
    Returns the LBP histogram as a feature vector.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    # Apply LBP
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    # Compute the histogram of LBP and normalize it
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize histogram to sum to 1
    
    return lbp_hist


# Paths to the image datasets
acc_folder_path = r"C:\dataset\acc_images"
nonacc_folder_path = r"C:\project\Dataset\non accident images"

all_img = []
labels = []

# Extract features for accident images
for img_name in os.listdir(acc_folder_path):
    img_path = os.path.join(acc_folder_path, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract LBP features
    lbp_features = extract_lbp_features(image)
    
    # Extract statistical features
    stat_features = calculate_image_statistics(image)
    
    # Combine LBP features and statistical features
    features = np.concatenate([lbp_features, stat_features])
    
    # Append to all_img and labels
    all_img.append(features)
    labels.append(1)  # Label 1 for accident images

# Extract features for non-accident images
for img_name in os.listdir(nonacc_folder_path):
    img_path = os.path.join(nonacc_folder_path, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract LBP features
    lbp_features = extract_lbp_features(image)
    
    # Extract statistical features
    stat_features = calculate_image_statistics(image)
    
    # Combine LBP features and statistical features
    features = np.concatenate([lbp_features, stat_features])
    
    # Append to all_img and labels
    all_img.append(features)
    labels.append(0)  # Label 0 for non-accident images

# Convert lists to numpy arrays
X = np.array(all_img)  # Features
y = np.array(labels)   # Labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Random Forest Classifier in this case)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)


# Print classification report
print(classification_report(y_test, y_pred))

# Function to load a user-selected image and predict
def predict_image(image_path):
    # Load and preprocess the selected image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract LBP features
    lbp_features = extract_lbp_features(image)
    
    # Extract statistical features
    stat_features = calculate_image_statistics(image)
    
    # Combine LBP features and statistical features
    features = np.concatenate([lbp_features, stat_features])
    
    # Predict using the trained classifier
    prediction = clf.predict([features])
    
    # Display the images (input and preprocessed)
    img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 4))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Preprocessed (Grayscale) Image
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.title("Preprocessed Image")
    plt.axis('off')
    
    # # Feature Extraction Values (Printed)
    # plt.subplot(1, 3, 3)
    # plt.axis('off')
    # plt.text(0, 0.9, f'LBP Features: {lbp_features[:5]}', fontsize=8)
    print(f'Statistics: Mean={stat_features[0]:.2f}, Std Dev={stat_features[2]:.2f}')
    print(lbp_features)
    plt.show()
    
    # Print the prediction result
    if prediction == 1:
        print("This is an accident image.")
        
        import pywhatkit as kit
        import datetime

        def send_whatsapp_alert(message, phone_number):
            """Function to send WhatsApp alert using pywhatkit."""
            try:
                now = datetime.datetime.now()
                hours = now.hour
                minutes = now.minute + 1  # Send message in the next minute

                kit.sendwhatmsg(phone_number, message, hours, minutes)
                print("WhatsApp Alert Sent Successfully!")
            except Exception as e:
                print(f"Error sending alert: {e}")

        # Example: Detecting an accident and sending an alert
        def accident_detected():
            accident_status = True  # Assume an accident is detected (replace with real detection logic)
            
            if accident_status:
                alert_message = "🚨 Accident Detected! Immediate attention required. 🚑"
                phone_number = "+918072975809"  # Replace with recipient's WhatsApp number
                send_whatsapp_alert(alert_message, phone_number)

        # Run the detection simulation
        accident_detected()

    else:
        print("This is a non-accident image.")
        
# # Initialize Tkinter
# root = Tk()
# root.withdraw()  # Hide the root window

# # Open file dialog
# file_path = askopenfilename(title="Select an Image", filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])

# print("Selected file:", file_path)       
# Function to ask user to select an image using file dialog
#def select_image():
 # root = tk.Tk()
 ##root.withdraw()  # Hide the root window
 #image_path = PhotoImage.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
 #if image_path:  # Only predict if the user selects a file
 #  predict_image(image_path)
 #else:
  #   print("No file selected.")                                                   
# Call the select_image function to open the file dialog and predict
#select_image()

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Accident", "Accident"], yticklabels=["Non-Accident", "Accident"])
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import skew
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def calculate_image_statistics(image):
    """
    This function computes the mean, variance, standard deviation, median,
    and skewness of the image pixel intensities.
    """
    pixels = image.flatten()

    mean = np.mean(pixels)
    variance = np.var(pixels)
    std_dev = np.std(pixels)
    median = np.median(pixels)
    skewness = skew(pixels)
    
    return [mean, variance, std_dev, median, skewness]

def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract Local Binary Pattern (LBP) features from a grayscale image.
    Returns the LBP histogram as a feature vector.
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    return lbp_hist

# Paths to the image datasets
acc_folder_path = r"C:\dataset\acc_images"
nonacc_folder_path = r"C:\project\Dataset\non accident images"

all_img = []
labels = []

# Extract features for accident images
for img_name in os.listdir(acc_folder_path):
    img_path = os.path.join(acc_folder_path, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    lbp_features = extract_lbp_features(image)
    stat_features = calculate_image_statistics(image)
    features = np.concatenate([lbp_features, stat_features])
    
    all_img.append(features)
    labels.append(1)  # Label 1 for accident images

# Extract features for non-accident images
for img_name in os.listdir(nonacc_folder_path):
    img_path = os.path.join(nonacc_folder_path, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    lbp_features = extract_lbp_features(image)
    stat_features = calculate_image_statistics(image)
    features = np.concatenate([lbp_features, stat_features])
    
    all_img.append(features)
    labels.append(0)  # Label 0 for non-accident images

# Convert lists to numpy arrays
X = np.array(all_img)  # Features
y = np.array(labels)   # Labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Classification Report
classification_rep = classification_report(y_test, y_pred)

# Function to process the uploaded image
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    lbp_features = extract_lbp_features(gray_image)
    stat_features = calculate_image_statistics(gray_image)
    
    features = np.concatenate([lbp_features, stat_features])
    
    # Predict using the trained classifier
    prediction = clf.predict([features])

    return gray_image, lbp_features, stat_features, prediction

# Streamlit app
def main():
    st.title("Accident Detection from Images")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Display original image
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        gray_image, lbp_features, stat_features, prediction = process_image(image)
        
        # Show grayscale image
        st.image(gray_image, caption="Grayscale Image", use_column_width=True, channels="GRAY")

        # Display LBP Features and Statistics
        st.write(f"LBP Features (first 5): {lbp_features[:5]}")
        st.write(f"Statistical Features: Mean={stat_features[0]:.2f}, Std Dev={stat_features[2]:.2f}")
        import pywhatkit as kit
        import datetime
        
        def send_whatsapp_alert(message, phone_number):
            """Function to send WhatsApp alert using pywhatkit."""
            try:
                now = datetime.datetime.now()
                hours = now.hour
                minutes = now.minute + 1  # Send message in the next minute
        
                kit.sendwhatmsg(phone_number, message, hours, minutes)
                print("WhatsApp Alert Sent Successfully!")
            except Exception as e:
                print(f"Error sending alert: {e}")
        
        # Example: Detecting an accident and sending an alert
        def accident_detected():
            accident_status = True  # Assume an accident is detected (replace with real detection logic)
            
            if accident_status:
                alert_message = "🚨 Accident Detected! Immediate attention required. 🚑"
                phone_number = "+918072975809"  # Replace with recipient's WhatsApp number
                send_whatsapp_alert(alert_message, phone_number)
                    
                    
        # Show the prediction
        if prediction == 1:
            # Run the detection simulation
            accident_detected()
            st.write("Prediction: Accident Image")
            

            

            
            
            
            
            
            
            
            
            
            
        else:
            st.write("Prediction: Non-Accident Image")
        
        # Display Classification Report
        st.subheader("Classification Report")
        st.text(classification_rep)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Accident", "Accident"], yticklabels=["Non-Accident", "Accident"], ax=ax)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

if __name__ == "__main__":
    main()

