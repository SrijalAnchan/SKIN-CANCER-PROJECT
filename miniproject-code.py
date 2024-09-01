import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image
import pyttsx3

# Function to handle login
def login():
    st.markdown("<h1 style='text-align: center; color: blue;'>Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state['logged_in'] = True
            st.success("Logged in successfully")
        else:
            st.error("Invalid credentials")

# Check login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
else:
    st.markdown("<h1 style='text-align: center; color: blue;'>Skin Cancer Classification</h1>", unsafe_allow_html=True)

    # Set up image data generators for loading and augmenting data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Update the path to your dataset directory
    train_generator = train_datagen.flow_from_directory(
        r"D:\MINIPROJECT2\train",  # Replace with your dataset path
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        r"D:\MINIPROJECT2\train",  # Use the same directory for validation
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )

    # Sidebar information
    st.sidebar.markdown("<h2 style='color: blue;'>Model Information</h2>", unsafe_allow_html=True)
    st.sidebar.write(f"Training samples: {train_generator.samples}")
    st.sidebar.write(f"Validation samples: {validation_generator.samples}")

    # Get the number of classes
    num_classes = len(train_generator.class_indices)
    st.sidebar.write(f"Number of classes: {num_classes}")

    # Build a simpler CNN model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Set steps per epoch and validation steps
    steps_per_epoch = max(1, 2 // 3)
    validation_steps = max(1, 2 // 3)

    st.sidebar.write(f"Steps per epoch: {steps_per_epoch}")
    st.sidebar.write(f"Validation steps: {validation_steps}")

    # Train the model for fewer epochs
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=5
    )

    # Plot training and validation accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='green')
    ax1.legend(loc='lower right')
    ax1.set_title('Training and Validation Accuracy')

    ax2.plot(epochs_range, loss, label='Training Loss', color='blue')
    ax2.plot(epochs_range, val_loss, label='Validation Loss', color='green')
    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation Loss')

    # Save the model
    model.save('skin_cancer_model_simplified.h5')

    # Streamlit UI
    st.markdown("<h2 style='text-align: center; color: blue;'>Upload an image of a skin lesion to classify its type</h2>", unsafe_allow_html=True)

    # Save and display plots in Streamlit
    st.pyplot(fig)

    # Function to predict an image
    def predict_image(img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        class_indices = train_generator.class_indices
        class_names = list(class_indices.keys())
        predicted_class = class_names[np.argmax(prediction)]
        predicted_probabilities = prediction[0]

        return predicted_class, predicted_probabilities

    # Symptoms for benign and malignant skin cancers
    benign_symptoms = """
    Benign skin cancers like basal cell carcinoma and squamous cell carcinoma often appear as new growths or patches with irregular borders and varying colors such as pink, red, or pearly. They can range in size and may have a textured surface, sometimes bleeding or itching. Consulting a dermatologist is crucial for accurate diagnosis and appropriate treatment.
    """

    malignant_symptoms = """
    Malignant skin cancers, like melanoma, often present as changes in moles with asymmetry, irregular borders, uneven color, or new growths that itch, bleed, or do not heal. They can also spread to nearby lymph nodes, causing swelling or lumps. Early detection through regular skin checks and prompt medical evaluation are essential for effective treatment outcomes.
    """

    # Function to speak the detected skin cancer type and symptoms
    def speak(text):
        engine = pyttsx3.init()
        # Adjust the speech rate
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 50)  # Decrease the rate by 50
        engine.say(text)
        engine.runAndWait()

    # Add a checkbox to enable or disable audio response
    enable_audio = st.sidebar.checkbox("Enable Audio Response", value=False)

    # Upload multiple images
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Save the uploaded file temporarily
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Predict the image
            predicted_class, predicted_probabilities = predict_image("temp.jpg")

            # Display the prediction
            st.markdown(f"<h3 style='color: green;'>The lesion is predicted to be: {predicted_class}</h3>", unsafe_allow_html=True)

            # Display the prediction probabilities
            st.markdown("<h4>Prediction Probabilities:</h4>", unsafe_allow_html=True)
            for class_name, prob in zip(list(train_generator.class_indices.keys()), predicted_probabilities):
                st.write(f"{class_name}: {prob * 100:.2f}%")

            # Display and speak the symptoms based on the prediction
            if predicted_class == 'benign':
                st.markdown(f"<h3 style='color: blue;'>Benign Symptoms:</h3>", unsafe_allow_html=True)
                st.write(benign_symptoms)
                if enable_audio:
                    speak(f"The lesion is predicted to be benign. Symptoms include: {benign_symptoms}")
            elif predicted_class == 'malignant':
                st.markdown(f"<h3 style='color: red;'>Malignant Symptoms:</h3>", unsafe_allow_html=True)
                st.write(malignant_symptoms)
                if enable_audio:
                    speak(f"The lesion is predicted to be malignant. Symptoms include: {malignant_symptoms}")

            # Plot confidence bar chart
            class_names = list(train_generator.class_indices.keys())
            fig, ax = plt.subplots()
            ax.bar(class_names, predicted_probabilities, color='blue')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities for Each Class')

            # Save and display the bar chart in Streamlit
            st.pyplot(fig)

    # Additional buttons for extra functionalities
    st.sidebar.header("Additional Options")

    if st.sidebar.button('View Model Summary'):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.sidebar.text("\n".join(model_summary))