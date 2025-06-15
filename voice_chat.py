# Copyright 2025 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

import pyaudio
import os
import json
import threading
import time
from deepgram import DeepgramClient, DeepgramClientOptions, AgentWebSocketEvents, AgentKeepAlive
from deepgram.clients.agent.v1.websocket.options import SettingsOptions
from dotenv import load_dotenv

load_dotenv()

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create a WAV header for audio data"""
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    
    header = bytearray(44)
    header[0:4] = b'RIFF'
    header[4:8] = b'\x00\x00\x00\x00'  # File size placeholder
    header[8:12] = b'WAVE'
    header[12:16] = b'fmt '
    header[16:20] = b'\x10\x00\x00\x00'  # Subchunk1Size (16 for PCM)
    header[20:22] = b'\x01\x00'  # AudioFormat (1 for PCM)
    header[22:24] = channels.to_bytes(2, 'little')  # NumChannels
    header[24:28] = sample_rate.to_bytes(4, 'little')  # SampleRate
    header[28:32] = byte_rate.to_bytes(4, 'little')  # ByteRate
    header[32:34] = block_align.to_bytes(2, 'little')  # BlockAlign
    header[34:36] = bits_per_sample.to_bytes(2, 'little')  # BitsPerSample
    header[36:40] = b'data'
    header[40:44] = b'\x00\x00\x00\x00'  # Subchunk2Size placeholder
    
    return header

def main():
    try:
        # Initialize Deepgram client
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")
        print("API Key found")

        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(api_key, config)
        connection = deepgram.agent.websocket.v("1")
        print("Created WebSocket connection...")

        # Configure the Agent
        options = SettingsOptions()
        options.audio.input.encoding = "linear16"
        options.audio.input.sample_rate = RATE
        options.audio.output.encoding = "linear16"
        options.audio.output.sample_rate = RATE
        options.audio.output.container = "wav"
        options.agent.language = "en"
        options.agent.listen.provider.type = "deepgram"
        options.agent.listen.provider.model = "nova-3"
        options.agent.think.provider.type = "open_ai"
        options.agent.think.provider.model = "gpt-4o-mini"
        options.agent.think.prompt = "You are a friendly AI assistant."
        options.agent.speak.provider.type = "deepgram"
        options.agent.speak.provider.model = "aura-2-thalia-en"
        options.agent.greeting = "Hello! How can I help you today?"

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        input_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

        # Audio buffer for agent responses
        audio_buffer = bytearray()
        file_counter = 0
        processing_complete = False

        # Send Keep Alive messages
        def send_keep_alive():
            while True:
                time.sleep(5)
                print("Keep alive!")
                connection.send(str(AgentKeepAlive()))

        keep_alive_thread = threading.Thread(target=send_keep_alive, daemon=True)
        keep_alive_thread.start()

        # Event Handlers
        def on_audio_data(self, data, **kwargs):
            nonlocal audio_buffer
            audio_buffer.extend(data)
            output_stream.write(data)  # Play audio in real-time
            print(f"Received audio data: {len(data)} bytes")

        def on_agent_audio_done(self, agent_audio_done, **kwargs):
            nonlocal audio_buffer, file_counter, processing_complete
            print("Agent audio done")
            if len(audio_buffer) > 0:
                with open(f"output-{file_counter}.wav", 'wb') as f:
                    f.write(create_wav_header())
                    f.write(audio_buffer)
                print(f"Saved output-{file_counter}.wav")
            audio_buffer = bytearray()
            file_counter += 1
            processing_complete = True

        def on_conversation_text(self, conversation_text, **kwargs):
            print(f"Conversation: {conversation_text}")
            with open("chatlog.txt", 'a') as chatlog:
                chatlog.write(f"{json.dumps(conversation_text.__dict__)}\n")

        def on_welcome(self, welcome, **kwargs):
            print(f"Welcome: {welcome}")
            with open("chatlog.txt", 'a') as chatlog:
                chatlog.write(f"Welcome: {welcome}\n")

        def on_settings_applied(self, settings_applied, **kwargs):
            print(f"Settings applied: {settings_applied}")

        def on_user_started_speaking(self, user_started_speaking, **kwargs):
            print(f"User started speaking: {user_started_speaking}")
            with open("chatlog.txt", 'a') as chatlog:
                chatlog.write(f"User started speaking: {user_started_speaking}\n")

        def on_agent_thinking(self, agent_thinking, **kwargs):
            print(f"Agent thinking: {agent_thinking}")

        def on_agent_started_speaking(self, agent_started_speaking, **kwargs):
            nonlocal audio_buffer
            audio_buffer = bytearray()
            print(f"Agent started speaking: {agent_started_speaking}")

        def on_close(self, close, **kwargs):
            print(f"Connection closed: {close}")
            with open("chatlog.txt", 'a') as chatlog:
                chatlog.write(f"Connection closed: {close}\n")

        def on_error(self, error, **kwargs):
            print(f"Error: {error}")
            with open("chatlog.txt", 'a') as chatlog:
                chatlog.write(f"Error: {error}\n")

        def on_unhandled(self, unhandled, **kwargs):
            print(f"Unhandled event: {unhandled}")

        # Register handlers
        connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
        connection.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        connection.on(AgentWebSocketEvents.Welcome, on_welcome)
        connection.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
        connection.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
        connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
        connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
        connection.on(AgentWebSocketEvents.Close, on_close)
        connection.on(AgentWebSocketEvents.Error, on_error)
        connection.on(AgentWebSocketEvents.Unhandled, on_unhandled)
        print("Event handlers registered")

        # Start the connection
        print("Starting WebSocket connection...")
        if not connection.start(options):
            print("Failed to start connection")
            return
        print("WebSocket connection started")

        # Stream microphone input
        print("Listening to microphone... Press Ctrl+C to stop")
        def stream_audio():
            try:
                while True:
                    data = input_stream.read(CHUNK, exception_on_overflow=False)
                    connection.send(data)
            except KeyboardInterrupt:
                pass

        audio_thread = threading.Thread(target=stream_audio, daemon=True)
        audio_thread.start()

        # Keep the program running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()
            connection.finish()
            print("Finished")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
