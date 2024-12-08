import socket
import pyaudio
import threading
import numpy as np
from scipy.signal import firwin, lfilter

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Set up client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("127.0.0.1", 5000)) # Replace with server's IP if necessary

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream for microphone capture and speaker playback
stream_input = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
stream_output = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# FIR Filter design for High-pass and Low-pass filters
def fir_highpass(cutoff, fs, numtaps=201, window='hamming'):
    taps = firwin(numtaps, cutoff, fs=fs, pass_zero=False, window=window)
    return taps

def fir_lowpass(cutoff, fs, numtaps=201, window='hamming'):
    taps = firwin(numtaps, cutoff, fs=fs, pass_zero=True, window=window)
    return taps

def apply_fir_filter(data, taps):
    return lfilter(taps, 1.0, data)

# Send audio to the server
def send_audio(filter_type='bandpass', cutoff1=300, cutoff2=3400, numtaps=201, window='hamming'):
    if filter_type == 'highpass':
        taps = fir_highpass(cutoff1, RATE, numtaps, window)
    elif filter_type == 'lowpass':
        taps = fir_lowpass(cutoff1, RATE, numtaps, window)
    else: # Default to bandpass
        taps = fir_bandpass(cutoff1, cutoff2, RATE, numtaps, window)

    while True:
        try:
            data = stream_input.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) # Convert to float32
            filtered_data = apply_fir_filter(audio_data, taps)
            filtered_data = np.clip(filtered_data, -32768, 32767) # Clip values
            filtered_data_byte = filtered_data.astype(np.int16).tobytes() # Convert back to int16
            client_socket.sendall(filtered_data_byte)
        except Exception as e:
            print(f"Error in sending audio: {e}")
            break

# Receive audio from the server and play it back
def receive_audio(filter_type='bandpass', cutoff1=300, cutoff2=3400, numtaps=201, window='hamming'):
    if filter_type == 'highpass':
        taps = fir_highpass(cutoff1, RATE, numtaps, window)
    elif filter_type == 'lowpass':
        taps = fir_lowpass(cutoff1, RATE, numtaps, window)
    else: # Default to bandpass
        taps = fir_bandpass(cutoff1, cutoff2, RATE, numtaps, window)

    while True:
        try:
            data = client_socket.recv(CHUNK)
            if data:
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) # Convert to float32
                filtered_data = apply_fir_filter(audio_data, taps)
                filtered_data = np.clip(filtered_data, -32768, 32767) # Clip values
                filtered_data_byte = filtered_data.astype(np.int16).tobytes() # Convert back to int16
                stream_output.write(filtered_data_byte)
        except Exception as e:
            print(f"Error in receiving audio: {e}")
            break

# Start sending and receiving in separate threads
filter_type = 'highpass' # Change to 'lowpass' or 'bandpass' as needed
cutoff1 = 300 # For highpass/lowpass
cutoff2 = 3400 # For bandpass
send_thread = threading.Thread(target=send_audio, args=(filter_type, cutoff1, cutoff2, 201, 'hamming'))
receive_thread = threading.Thread(target=receive_audio, args=(filter_type, cutoff1, cutoff2, 201, 'hamming'))
send_thread.start()
receive_thread.start()

send_thread.join()
receive_thread.join()

# Close sockets and streams
client_socket.close()
stream_input.stop_stream()
stream_input.close()
stream_output.stop_stream()
stream_output.close()
audio.terminate()
