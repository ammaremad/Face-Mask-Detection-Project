```markdown  
# Face Mask Detection Project  

## Overview  

This project implements a face mask detection system using deep learning techniques. It can identify whether people are wearing masks in images and real-time video streams. The project utilizes a dataset of 12,000 images and explores different CNN architectures and transfer learning approaches to achieve high accuracy.  

## Dataset  

The dataset used in this project is the "Face Mask 12K Images Dataset" from Kaggle:  

*   **URL:** [https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)  
*   **License:** CC0-1.0  

The dataset contains 12,000 images divided into two categories: "with mask" and "without mask." The dataset is split into 10,000 images for training and 800 for validation.  

## Models  

The project explores the following models:  

*   **Custom CNN:** A convolutional neural network built from scratch using Keras . It consists of Conv2D, MaxPooling2D, Flatten, and Dense layers. Achieved approximately 98.56% training accuracy and 99.25% validation accuracy.  
*   **ResNet50:** A pre-trained ResNet50 model fine-tuned for face mask detection. Achieved lower accuracy compared to other models (68.75%).  
*   **MobileNetV2:** A pre-trained MobileNetV2 model fine-tuned for face mask detection. This model is preferred for real-time applications due to its lighter architecture. Achieved 99.50% validation accuracy.  
*   **VGG16:** A pre-trained VGG16 model fine-tuned for face mask detection. Achieved 99.75% validation accuracy.  

## Dependencies  

*   tensorflow  
*   keras  
*   opencv-python  
*   numpy  
*   matplotlib  
*   pandas  
*   tkinter  

Install the necessary packages using pip:  

```bash  
pip install tensorflow opencv-python numpy matplotlib pandas  
```  

## Usage  

1.  **Data Preparation:**  

    *   Download the dataset from Kaggle and unzip it.  
    *   Organize the data into training and validation directories. The `ImageDataGenerator` is used to prepare the data. Example paths used in the project: `r'C:\Users\aliem\Downloads\archive (3)\Face Mask Dataset\Train'` and `r'C:\Users\aliem\Downloads\archive (3)\Face Mask Dataset\Validation'`.  

2.  **Model Training:**  

    *   The CNN model can be built and trained using the code in the `Build CNN Model` section. The model is compiled using the Adam optimizer and binary cross-entropy loss.  
    *   Transfer learning models (ResNet50, MobileNetV2, VGG16) can be built and trained using the `build_model` function.  

3.  **Model Evaluation:**  

    *   Evaluate the trained models on the validation set to compare their performance.  

4.  **Testing the Model:**  

    *   Load the best-performing model (`best_model.h5`).  
    *   Load the face detector: `face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')`.  
    *   Use tkinter to select image files for testing.  
    *   The code in the `Test the Model` section can be used to test the model on image files.  

    ```python  
    import cv2  
    import numpy as np  
    from tensorflow.keras.models import load_model  
    from tkinter import Tk, filedialog  
    import matplotlib.pyplot as plt  

    # Load the trained model  
    model = load_model('best_model.h5')  

    # Load the face detector  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

    # Use tkinter to select image files  
    root = Tk()  
    root.withdraw()  # Hide the main window  
    file_paths = filedialog.askopenfilenames(title="Select one or more images", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])  

    # Process each selected image  
    for file_path in file_paths:  
        img = cv2.imread(file_path)  
        if img is None:  
            print(f"Failed to load: {file_path}")  
            continue  

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  

        for (x, y, w, h) in faces:  
            face = img[y:y+h, x:x+w]  
            face_resized = cv2.resize(face, (224, 224)) / 255.0  
            face_input = np.expand_dims(face_resized, axis=0)  
            pred = model.predict(face_input)[0][0]  
            label = "With Mask" if pred < 0.5 else "No Mask"  
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)  
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)  
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  

        print(f"Processed image: {file_path}")  
        # Convert BGR to RGB for displaying in Jupyter  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        # Display the image in Jupyter Notebook  
        plt.figure(figsize=(10, 6))  
        plt.imshow(img_rgb)  
        plt.axis('off')  
        plt.title('Face_Mask_Detection')  
        plt.show()  
    ```  

5.  **Real-time Face Mask Detection:**  

    *   Load the pre-trained face mask detection model: `model = load_model("best_model.h5")`.  
    *   Load the OpenCV DNN face detector: `face_net = cv2.dnn.readNetFromCaffe( prototxt="deploy.prototxt.txt", caffeModel="res10_300x300_ssd_iter_140000.caffemodel")`.  Ensure the paths to `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel` are correct.  
    *   The code in the `Realtime face mask Detection` section can be used to run real-time face mask detection using the webcam.  

    ```python  
    import cv2  
    import numpy as np  
    from tensorflow.keras.models import load_model  

    # Load mask classification model  
    model = load_model("best_model.h5")  

    # Load OpenCV DNN face detector  
    face_net = cv2.dnn.readNetFromCaffe(  
        prototxt="deploy.prototxt.txt",  
        caffeModel="res10_300x300_ssd_iter_140000.caffemodel"  
    )  

    def predict_mask(face_img):  
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  
        resized = cv2.resize(face_rgb, (224, 224)) / 255.0  
        resized = np.expand_dims(resized, axis=0)  
        pred = model.predict(resized, verbose=0)[0][0]  
        confidence = (1 - pred) if pred < 0.5 else pred  
        label = f"{'With Mask' if pred < 0.5 else 'No Mask'}: {confidence:.2f}"  
        color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)  
        return label, color  

    def start_camera():  
        cap = cv2.VideoCapture(0)  
        while True:  
            ret, frame = cap.read()  
            if not ret:  
                break  

            h, w = frame.shape[:2]  
            # Prepare the frame for DNN face detector  
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, crop=False)  
            face_net.setInput(blob)  
            detections = face_net.forward()  

            for i in range(detections.shape[2]):  
                confidence = detections[0, 0, i, 2]  
                if confidence > 0.6:  
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  
                    (x1, y1, x2, y2) = box.astype("int")  
                    face = frame[y1:y2, x1:x2]  
                    if face.size == 0:  
                        continue  

                    label, color = predict_mask(face)  
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  

            cv2.imshow("MaskDetection", frame)  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  

        cap.release()  
        cv2.destroyAllWindows()  

    start_camera()  
    ```  

## Results  

The project achieved the following results:  

*   **CNN Model:** Approximately 98.56% training accuracy and 99.25% validation accuracy.  
*   **ResNet50:** Lower accuracy compared to other models (68.75%).  
*   **MobileNetV2:** 99.50% validation accuracy.  
*   **VGG16:** 99.75% validation accuracy.  

MobileNetV2 is preferred for real-time applications due to its balance of accuracy and efficiency.  

## Conclusion  

This project demonstrates the effectiveness of deep learning techniques for face mask detection. The MobileNetV2 model offers a good balance between accuracy and computational efficiency, making it suitable for real-time applications.  
```
