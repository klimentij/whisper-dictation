import argparse
import time
import threading
import pyaudio
import numpy as np
import rumps
from pynput import keyboard
import platform
import os
import tempfile
import wave
import subprocess

class SubprocessTranscriber:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pykeyboard = keyboard.Controller()
        self.whisper_path = "../whisper.cpp/main"
        self.model_path = f"../whisper.cpp/models/ggml-{model_name}.bin"
        
        # Create audio directory if it doesn't exist
        os.makedirs('audio', exist_ok=True)
        
        # Verify whisper executable exists
        if not os.path.exists(self.whisper_path):
            raise RuntimeError(f"Whisper executable not found at {self.whisper_path}")
        
        # Verify model exists
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model not found at {self.model_path}")
        
        print(f"Using model: {self.model_path}")

    def transcribe(self, audio_data, language=None):
        try:
            # Create timestamped filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            wav_path = os.path.join('audio', f'recording_{timestamp}.wav')
            
            # Save WAV file
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())
            
            # Create temporary copy for whisper processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                # Copy the saved WAV file to temp location
                with open(wav_path, 'rb') as f:
                    temp_wav.write(f.read())
            
            # Build command with proper arguments
            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", temp_wav.name,
                "-pp",           # print progress
                "-otxt",         # output text format
                "--output-file", temp_wav.name,  # base name for output
                "-t", "4",       # number of threads
                "-pc"           # print colors
            ]
            
            if language:
                cmd.extend(["-l", language])
            else:
                cmd.extend(["-l", "auto"])  # auto-detect language if none specified
            
            # Run whisper.cpp
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Read the output text file
                output_txt = temp_wav.name + ".txt"
                if os.path.exists(output_txt):
                    with open(output_txt, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                else:
                    text = result.stdout.strip()
                
                # Type out the text
                for char in text:
                    try:
                        self.pykeyboard.type(char)
                        time.sleep(0.0025)
                    except:
                        pass
                        
            finally:
                # Only delete temp files, keep the saved WAV
                os.unlink(temp_wav.name)
                if os.path.exists(temp_wav.name + ".txt"):
                    os.unlink(temp_wav.name + ".txt")
                    
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()

    def __del__(self):
        pass  # No cleanup needed

class Recorder:
    def __init__(self, transcriber):
        self.recording = False
        self.transcriber = transcriber

    def start(self, language=None):
        thread = threading.Thread(target=self._record_impl, args=(language,))
        thread.start()

    def stop(self):
        self.recording = False


    def _record_impl(self, language):
        self.recording = True
        frames_per_buffer = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        frames_per_buffer=frames_per_buffer,
                        input=True)
        frames = []

        while self.recording:
            data = stream.read(frames_per_buffer)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0
        self.transcriber.transcribe(audio_data_fp32, language)


class GlobalKeyListener:
    def __init__(self, app, key_combination):
        self.app = app
        self.key1, self.key2 = self.parse_key_combination(key_combination)
        self.key1_pressed = False
        self.key2_pressed = False

    def parse_key_combination(self, key_combination):
        key1_name, key2_name = key_combination.split('+')
        key1 = getattr(keyboard.Key, key1_name, keyboard.KeyCode(char=key1_name))
        key2 = getattr(keyboard.Key, key2_name, keyboard.KeyCode(char=key2_name))
        return key1, key2

    def on_key_press(self, key):
        if key == self.key1:
            self.key1_pressed = True
        elif key == self.key2:
            self.key2_pressed = True

        if self.key1_pressed and self.key2_pressed:
            self.app.toggle()

    def on_key_release(self, key):
        if key == self.key1:
            self.key1_pressed = False
        elif key == self.key2:
            self.key2_pressed = False

class DoubleCommandKeyListener:
    def __init__(self, app):
        self.app = app
        self.key = keyboard.Key.cmd_r
        self.pressed = 0
        self.last_press_time = 0

    def on_key_press(self, key):
        is_listening = self.app.started
        if key == self.key:
            current_time = time.time()
            if not is_listening and current_time - self.last_press_time < 0.5:  # Double click to start listening
                self.app.toggle()
            elif is_listening:  # Single click to stop listening
                self.app.toggle()
            self.last_press_time = current_time

    def on_key_release(self, key):
        pass

class StatusBarApp(rumps.App):
    def __init__(self, recorder, languages=None, max_time=None):
        super().__init__("whisper", "⏯")
        self.languages = languages
        self.current_language = languages[0] if languages is not None else None

        menu = [
            'Start Recording',
            'Stop Recording',
            None,
        ]

        if languages is not None:
            for lang in languages:
                callback = self.change_language if lang != self.current_language else None
                menu.append(rumps.MenuItem(lang, callback=callback))
            menu.append(None)
            
        self.menu = menu
        self.menu['Stop Recording'].set_callback(None)

        self.started = False
        self.recorder = recorder
        self.max_time = max_time
        self.timer = None
        self.elapsed_time = 0

    def change_language(self, sender):
        self.current_language = sender.title
        for lang in self.languages:
            self.menu[lang].set_callback(self.change_language if lang != self.current_language else None)

    @rumps.clicked('Start Recording')
    def start_app(self, _):
        print('Listening...')
        self.started = True
        self.menu['Start Recording'].set_callback(None)
        self.menu['Stop Recording'].set_callback(self.stop_app)
        self.recorder.start(self.current_language)

        if self.max_time is not None:
            self.timer = threading.Timer(self.max_time, lambda: self.stop_app(None))
            self.timer.start()

        self.start_time = time.time()
        self.update_title()

    @rumps.clicked('Stop Recording')
    def stop_app(self, _):
        if not self.started:
            return
        
        if self.timer is not None:
            self.timer.cancel()

        print('Transcribing...')
        self.title = "⏯"
        self.started = False
        self.menu['Stop Recording'].set_callback(None)
        self.menu['Start Recording'].set_callback(self.start_app)
        self.recorder.stop()
        print('Done.\n')

    def update_title(self):
        if self.started:
            self.elapsed_time = int(time.time() - self.start_time)
            minutes, seconds = divmod(self.elapsed_time, 60)
            self.title = f"({minutes:02d}:{seconds:02d}) 🔴"
            threading.Timer(1, self.update_title).start()

    def toggle(self):
        if self.started:
            self.stop_app(None)
        else:
            self.start_app(None)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dictation app using whisper.cpp ASR model. By default the keyboard shortcut cmd+option '
        'starts and stops dictation')
    parser.add_argument('-m', '--model_name', type=str,
                        choices=['tiny', 'tiny-q5_1', 'tiny.en', 'tiny.en-q5_1', 'tiny.en-q8_0',
                                'base', 'base-q5_1', 'base.en', 'base.en-q5_1',
                                'small', 'small-q5_1', 'small.en', 'small.en-q5_1',
                                'medium', 'medium-q5_0', 'medium.en', 'medium.en-q5_0',
                                'large-v1', 'large-v2', 'large-v2-q5_0',
                                'large-v3', 'large-v3-q5_0', 'large-v3-turbo'],
                        default='large-v3-turbo',
                        help='Specify the whisper.cpp model to use. Check https://github.com/ggerganov/whisper.cpp for available models.')
    parser.add_argument('-k', '--key_combination', type=str, default='cmd_l+alt' if platform.system() == 'Darwin' else 'ctrl+alt',
                        help='Specify the key combination to toggle the app. Example: cmd_l+alt for macOS '
                        'ctrl+alt for other platforms. Default: cmd_r+alt (macOS) or ctrl+alt (others).')
    parser.add_argument('--k_double_cmd', action='store_true',
                            help='If set, use double Right Command key press on macOS to toggle the app (double click to begin recording, single click to stop recording). '
                                 'Ignores the --key_combination argument.', default=True)
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Specify the two-letter language code (e.g., "en" for English) to improve recognition accuracy. '
                        'This can be especially helpful for smaller model sizes.  To see the full list of supported languages, '
                        'check out the official list [here](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py).')
    parser.add_argument('-t', '--max_time', type=float, default=30,
                        help='Specify the maximum recording time in seconds. The app will automatically stop recording after this duration. '
                        'Default: 30 seconds.')

    args = parser.parse_args()

    if args.language is not None:
        args.language = args.language.split(',')

    if args.model_name.endswith('.en') and args.language is not None and any(lang != 'en' for lang in args.language):
        raise ValueError('If using a model ending in .en, you cannot specify a language other than English.')

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Initializing transcriber...")
    transcriber = SubprocessTranscriber(args.model_name)
    recorder = Recorder(transcriber)
    
    app = StatusBarApp(recorder, args.language, args.max_time)
    if args.k_double_cmd:
        key_listener = DoubleCommandKeyListener(app)
    else:
        key_listener = GlobalKeyListener(app, args.key_combination)
    listener = keyboard.Listener(on_press=key_listener.on_key_press, on_release=key_listener.on_key_release)
    listener.start()

    print("Running... ")
    app.run()

