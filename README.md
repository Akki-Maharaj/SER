SPEECH EMOTION RECOGNITION (SER)
================================

ABOUT
-----
This repository implements a Speech Emotion Recognition (SER) system to classify the emotional state of a speaker from audio recordings. The goal is to demonstrate how to process raw audio, extract useful features, train a machine learning model, and evaluate its performance.

The SER model is built using the RAVDESS dataset and basic audio signal processing with machine learning techniques. It is a great starting point for students and researchers interested in affective computing and audio classification tasks.

TECHNOLOGIES AND LIBRARIES USED
-------------------------------
- Python 3.x
- NumPy : for numerical computations
- Pandas : for data manipulation and analysis
- Librosa : for audio signal processing and feature extraction
- Matplotlib and Seaborn : for data visualization
- Scikit-learn : for building and evaluating machine learning models
- RAVDESS Dataset : Ryerson Audio-Visual Database of Emotional Speech and Song

WORKFLOW / APPROACH
-------------------
1. DATA COLLECTION
   - Load audio samples from the RAVDESS dataset.
   - Each audio file is labeled with an emotion such as calm, happy, sad, angry, fearful, etc.

2. FEATURE EXTRACTION
   - Use Librosa to extract relevant acoustic features:
     - MFCC (Mel Frequency Cepstral Coefficients)
     - Chroma Frequencies
     - Mel Spectrogram
   - These features are combined into a single feature vector for each audio file.

3. DATA PREPROCESSING
   - Combine features into a Pandas DataFrame.
   - Encode categorical emotion labels into numerical values.
   - Split the dataset into training and testing sets.

4. MODEL BUILDING
   - Train a Multilayer Perceptron (MLP) classifier using Scikit-learn.
   - Fit the model to the training data.
   - Predict on the test data.

5. EVALUATION
   - Evaluate the model performance using accuracy score, confusion matrix, and classification report.
   - Visualize the results with heatmaps and plots.

HOW TO RUN
----------
1. Clone this repository.
2. Download the RAVDESS dataset and place it in the appropriate folder structure.
3. Install the required Python libraries using pip:
   pip install numpy pandas librosa matplotlib seaborn scikit-learn
4. Open the SER.ipynb notebook in Jupyter Notebook or Google Colab.
5. Run all cells step-by-step to execute the workflow.

CITATIONS
---------
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE, 13(5), e0196391.
- McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.

ACKNOWLEDGEMENTS
----------------
This project is inspired by open-source SER tutorials and implementations shared by the community.

For any issues or contributions, feel free to create a pull request or raise an issue on GitHub.

HAPPY CODING! 🎉
