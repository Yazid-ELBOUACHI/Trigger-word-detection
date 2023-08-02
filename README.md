# Audio Trigger Word Detection

This repository contains code for detecting a trigger word ("activate") in audio clips using a deep learning model. The model is trained to predict the presence of the trigger word in spectrograms generated from audio recordings. The spectrograms are computed to represent the frequency content of the audio over time, making it easier for the model to learn the patterns associated with the trigger word.

## Packages Used
- NumPy: For numerical computing in Python
- PyDub: For audio processing, loading, and saving audio files
- IPython: For audio playback and visualization
- TensorFlow and Keras: For building and training the deep learning model

## Data Synthesis
To train the model, we synthesize a dataset by combining background audio clips with random occurrences of the trigger word ("activate") and negative words. This approach allows us to generate a large amount of labeled training data without manually labeling each example.

## Spectrogram
The raw audio recordings are converted into spectrograms, which represent how different frequencies are present in the audio over time. This representation makes it easier for the model to learn to detect the trigger word.

## Model Architecture
The deep learning model consists of a Convolutional Neural Network (CNN) followed by two Gated Recurrent Units (GRUs) and a TimeDistributed dense layer. The model takes as input a spectrogram of shape `(Tx, n_freq)` and outputs a prediction for each time step in the spectrogram.

## Training and Evaluation
The model is trained using binary cross-entropy loss and the Adam optimizer. The training dataset is synthesized from background audio, trigger word clips, and negative word clips. The accuracy on the development set is used to evaluate the model's performance.

## Detection and Chime
The trained model can be used to detect the trigger word in any audio clip. When the trigger word is detected in the audio, a chime sound is superimposed on the audio to indicate the presence of the trigger word.

## How to Use
1. Install the required packages mentioned in the "Packages Used" section.
2. Download the dataset of background audio, trigger words ("activate") clips, and negative words (not "activate").
3. Run the script to synthesize the training examples and train the model.
4. Optionally, you can fine-tune the pre-trained model by blocking the weights of the batch normalization layers.
5. Test the model on the development set using the `model.evaluate()` function.
6. Make predictions on audio clips using the `detect_triggerword()` function. The function will display the spectrogram and the predictions, and if the trigger word is detected, a chime sound will be superimposed on the audio.

Please note that this repository is provided as an example, and you may need to customize the code according to your specific audio dataset and requirements.

Happy trigger word detection!
