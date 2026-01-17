 # 🍌 Banana Shelf-Life Predictor (Regression AI)
 
📌 Project Overview
This project implements an end-to-end machine learning pipeline designed to mitigate food waste by quantifying produce ripeness. Unlike standard classification models (which only label "Fresh" vs "Rotten"), this system uses Computer Vision Regression to predict the exact number of days remaining until a banana becomes overripe.

-- It highlights the business value (food waste) and the technical challenge (regression + small data).

The system was trained on a custom-curated temporal dataset and utilizes Transfer Learning (ResNet-18) to achieve high accuracy even with limited data samples.

🚀 Key Features
Regression vs. Classification: Predicts a continuous value (e.g., "2.4 days left") rather than discrete classes.

Data-Efficient Learning: Achieved a Mean Absolute Error (MAE) of ~0.57 days using aggressive data augmentation strategies (rotation, lighting noise) to overcome data scarcity.

End-to-End Deployment: Full-stack implementation from raw data collection to a production-ready Streamlit web application.

Model Architecture: Fine-tuned ResNet-18 with a custom regression head.

🛠️ Tech Stack
Core: Python 3.9+

Deep Learning: PyTorch, Torchvision (ResNet-18)

Data Processing: Pandas, Pillow, NumPy

Deployment: Streamlit Community Cloud

3. Quick-Start Command (Optional addition)
Add this to help users run it locally.

💻 How to Run Locally
Bash

# Clone the repository
git clone https://github.com/sunilkumar-nallani/Banana_Net.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
