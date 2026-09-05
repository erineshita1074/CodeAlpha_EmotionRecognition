# Emotion Recognition from Speech

## CodeAlpha Machine Learning Internship — Task 2

## Objective
Recognize human emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech audio using deep learning and signal processing techniques.

## Dataset
**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- Speech audio clips from 24 professional actors, 8 emotion categories
- Source: https://zenodo.org/record/1188976

## Approach
1. Downloaded and organized RAVDESS speech audio files by emotion label (parsed from filename convention)
2. Extracted MFCC (Mel-Frequency Cepstral Coefficients) features from each audio clip using Librosa
3. Encoded emotion labels and scaled feature vectors
4. Split data into train (80%) and test (20%) sets with stratification
5. Built a Dense Neural Network classifier for 8-class emotion classification
6. Evaluated using accuracy, confusion matrix, and per-class classification report

## Key Features
- Audio signal processing (MFCC feature extraction)
- Deep learning for multi-class audio classification
- Handling of imbalanced/multi-class emotion labels

## Results
- **Test Accuracy: 67%** (well above the 12.5% random-guess baseline for 8 balanced classes)

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.93 | 0.66 | 0.77 |
| Calm | 0.72 | 0.76 | 0.74 |
| Disgust | 0.59 | 0.61 | 0.60 |
| Fearful | 0.65 | 0.92 | 0.77 |
| Happy | 0.77 | 0.62 | 0.69 |
| Neutral | 0.33 | 0.37 | 0.35 |
| Sad | 0.69 | 0.58 | 0.63 |
| Surprised | 0.65 | 0.72 | 0.68 |

Weighted Avg F1-Score: 0.67. The model performs strongest on "fearful" and "angry" emotions, while "neutral" is the most challenging class to distinguish, likely due to its acoustic similarity to calm and sad speech patterns.

## Tech Stack
Python, TensorFlow/Keras, Librosa, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

## How to Run
```bash
pip install librosa soundfile tensorflow scikit-learn pandas matplotlib seaborn
```
Open `CodeAlpha_EmotionRecognition.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially. The notebook automatically downloads the RAVDESS dataset from Zenodo.

## Project Structure
- CodeAlpha_EmotionRecognition.ipynb
- README.md

## Author
Eshita — Final Year Software Engineering Student, Daffodil International University
