from gtts import gTTS
import os

# Define the message you want to convert to audio
message = "Fatigue pattern detected. It's dangerous to continue, please rest for a while!"

# Create a gTTS object
tts = gTTS(text=message, lang='en')

# Save the audio to a file (e.g., 'output.mp3')
tts.save('output.mp3')

# Play the audio (you can use other methods to play the audio as well)
os.system('afplay output.mp3')  # Use 'mpg321' on Linux or 'afplay' on macOS
