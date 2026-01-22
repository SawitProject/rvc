import os
import sys
import torch
import shutil
import threading
import subprocess

import tkinter as tk
import customtkinter as ctk

from tkinter import filedialog

sys.path.append(os.getcwd())

from rvc.lib.backend import opencl
from rvc.lib.config import Config
from rvc.infer.infer import run_inference_script

audio_load = ""

opencl_available = False
cpu_mode = False
is_half = False

config = Config(is_half=is_half, cpu_mode=cpu_mode)

if config.device.startswith("cpu"): print("[WARNING] No GPU found. Will run with CPU, inference will be very slow")
print(f"[INFO] Device: {config.device} Is_Half: {config.is_half}")

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

f0_method_full_list = [
    "pm", 
    "dio", 
    "mangio-crepe-tiny", 
    "mangio-crepe-small", 
    "mangio-crepe-medium", 
    "mangio-crepe-large", 
    "mangio-crepe-full", 
    "crepe-tiny", 
    "crepe-small", 
    "crepe-medium", 
    "crepe-large", 
    "crepe-full", 
    "fcpe", 
    "fcpe-legacy", 
    "rmvpe", 
    "rmvpe-legacy", 
    "harvest", 
    "yin", 
    "pyin", 
    "swipe"
]

embedders_full_list = [
    "contentvec_base", 
    "hubert_base", 
    "japanese_hubert_base", 
    "korean_hubert_base", 
    "chinese_hubert_base", 
    "portuguese_hubert_base"
]

model_folders = sorted([os.path.join(root, name) for root, _, files in os.walk("rvc_models", topdown=False) for name in files if name.endswith(".pth")])

def process_output(file_path):
    if not os.path.exists(file_path): return file_path
    file = os.path.splitext(os.path.basename(file_path))

    index = 1
    while 1:
        file_path = os.path.join(os.path.dirname(file_path), f"{file[0]}_RVC_{index}{file[1]}")
        if not os.path.exists(file_path): return file_path
        index += 1

def browse_file():
    filepath = filedialog.askopenfilename(
        filetypes=[
            (
                "Audio Files", "*.wav; *.mp3; *.flac; *.ogg; *.opus; *.m4a; *.mp4; *.aac; *.alac; *.wma; *.aiff; *.webm; *.ac3"
            )
        ]
    )

    filepath = os.path.normpath(filepath)

    input_path_frame.delete("0.0", tk.END)
    input_path_frame.insert("0.0", filepath)

    output_path_frame.delete("0.0", tk.END)
    output_path_frame.insert("0.0", process_output(filepath))

def move_files_from_directory(src_dir, dest_models, model_name):
    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".index"):
                filepath = os.path.join(dest_models, file.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip())

                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = os.path.join(dest_models, model_name + ".pth")

                shutil.move(file_path, pth_path)

def save_drop_model(dropbox):
    model_folders = "rvc_models" 
    save_model_temp = "save_model_temp"

    if not os.path.exists(model_folders): os.makedirs(model_folders, exist_ok=True)
    if not os.path.exists(save_model_temp): os.makedirs(save_model_temp, exist_ok=True)

    shutil.copy(dropbox, save_model_temp)

    try:
        print("[INFO] Start uploading...")
        file_name = os.path.basename(dropbox)
        model_folders = os.path.join(model_folders, file_name.replace(".zip", ""))

        os.makedirs(model_folders, exist_ok=True)
        shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)

        move_files_from_directory(save_model_temp, model_folders, file_name.replace(".zip", ""))
        print("[INFO] Completed upload.")
    except Exception as e:
        print(f"[ERROR] An error occurred during unpack: {e}")
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)

def update_model_value():
    global model_folders

    model_folders = sorted([os.path.join(root, name) for root, _, files in os.walk("rvc_models", topdown=False) for name in files if name.endswith(".pth")])
    model_path_dropdown.configure(values=model_folders)
    model_path_dropdown.update()

def index_select(choice):
    dirs = os.path.dirname(choice)

    for f in os.listdir(dirs):
        if f.endswith(".index") and "trained" not in f: 
            index_path_textbox.delete("0.0", tk.END)
            index_path_textbox.insert("0.0", os.path.join(dirs, f))
            return

    index_path_textbox.delete("0.0", tk.END)
    index_path_textbox.insert("0.0", "")

def browse_zip():
    global zip_file

    zip_file = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        filetypes=[
            (
                "Zip Files", "*.zip"
            )
        ],
    )
    save_drop_model(zip_file)
    update_model_value()

def pitch_slider_event(value):
    pitch_label.configure(text=f"PITCH: {round(value)}")

def index_strength_event(value):
    index_strength_label.configure(text=f"Index Strength: {round(value, 2)}")

def protect_event(value):
    protect_label.configure(text=f"Protecting: {round(value, 2)}")

def rms_mix_rate_event(value):
    rms_mix_rate_label.configure(text=f"RMS Mix Rate: {round(value, 1)}")

def hop_length_event(value):
    hop_length_label.configure(text=f"Hop Length: {round(value * 64)}")

def clean_strength_event(value):
    clean_strength_label.configure(text=f"Clean Strength: {round(value, 1)}")

def autotune_strength_event(value):
    autotune_strength_label.configure(text=f"AutoTune Strength: {round(value, 1)}")

def formant_qfrency_event(value):
    formant_qfrency_label.configure(text=f"Formant Qfrency: {round(value, 1)}")

def formant_timbre_event(value):
    formant_timbre_label.configure(text=f"Formant Timbre: {round(value, 1)}")

def proposal_pitch_threshold_event(value):
    proposal_pitch_threshold_label.configure(text=f"Proposal Pitch: {round(value, 1)}")

def refresh_slider_positions():
    y = 220

    show_hop_length_method = [
        "mangio-crepe-tiny", 
        "mangio-crepe-small", 
        "mangio-crepe-medium", 
        "mangio-crepe-large", 
        "mangio-crepe-full", 
        "fcpe", 
        "fcpe-legacy", 
        "yin", 
        "pyin"
    ]

    f0_value = f0_method_dropdown.get()

    if f0_value in show_hop_length_method:
        hop_length_label.place(x=95, y=y)
        y += 20

        hop_length_slider.place(x=15, y=y)
        y += 20
    else:
        hop_length_label.place_forget()
        hop_length_slider.place_forget()

    if clean_audio_checkbox.get():
        clean_strength_label.place(x=95, y=y)
        y += 20

        clean_strength_slider.place(x=15, y=y)
        y += 20
    else:
        clean_strength_label.place_forget()
        clean_strength_slider.place_forget()

    if autotune_checkbox.get():
        autotune_strength_label.place(x=95, y=y)
        y += 20

        autotune_strength_slider.place(x=15, y=y)
        y += 20
    else:
        autotune_strength_label.place_forget()
        autotune_strength_slider.place_forget()

    if formant_shifting_checkbox.get():
        formant_qfrency_label.place(x=95, y=y)
        y += 20

        formant_qfrency_slider.place(x=15, y=y)
        y += 20

        formant_timbre_label.place(x=95, y=y)
        y += 20

        formant_timbre_slider.place(x=15, y=y)
        y += 20
    else:
        formant_qfrency_label.place_forget()
        formant_qfrency_slider.place_forget()
        formant_timbre_label.place_forget()
        formant_timbre_slider.place_forget()

    if proposal_pitch_checkbox.get():
        proposal_pitch_threshold_label.place(x=95, y=y)
        y += 20

        proposal_pitch_threshold_slider.place(x=15, y=y)
        y += 20
    else:
        proposal_pitch_threshold_label.place_forget()
        proposal_pitch_threshold_slider.place_forget()

def update_config(selected):
    global config, cpu_mode, is_half, opencl_available

    if selected == "GPU":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif opencl.is_available() or opencl_available:
            device = "ocl:0"
            is_half = False

            opencl.torch_available = True
            opencl_available = False
        elif torch.backends.mps.is_available():
            device = "mps"
            is_half = False
    else:
        device = "cpu"
        is_half = False

        if opencl.is_available(): 
            opencl.torch_available = False
            opencl_available = True

    config.device, config.is_half = device, is_half
    print(f"[INFO] Device: {config.device} Is_Half: {config.is_half}")

    is_half_checkbox.configure(state=tk.NORMAL if config.device.startswith("cuda") else tk.DISABLED)
    is_half_checkbox.update()

def update_is_half():
    global config, cpu_mode, is_half

    is_half = bool(is_half_checkbox.get()) and config.device.startswith("cuda") and not cpu_mode
    config.is_half = is_half

    print(f"[INFO] Device: {config.device} Is_Half: {config.is_half}")

def play_audio():
    if os.path.exists(audio_load):
        audio_file = os.path.abspath(audio_load)

        if sys.platform == 'win32': subprocess.call(['start', '', audio_file], shell=True)
        elif sys.platform == 'darwin': subprocess.call(['open', audio_file])
        elif sys.platform == 'linux': subprocess.call(['xdg-open', audio_file])

def on_click():
    global audio_load

    result_state.place_forget()
    process_frame.place_forget()
    open_output_button.place_forget()

    try:
        pitch = round(pitch_slider.get())
        index_rate = round(index_strength_slider.get(), 2)
        volume_envelope = round(rms_mix_rate_slider.get(), 1)
        protect = round(protect_slider.get(), 2)
        hop_length = round(hop_length_slider.get() * 64)
        f0_method = f0_method_dropdown.get()
        input_path = input_path_frame.get("0.0", tk.END).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        output_path = (output_path_frame.get("0.0", tk.END) or process_output(input_path)).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        pth_path = model_path_dropdown.get().strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        index_path = index_path_textbox.get("0.0", tk.END).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        export_format = os.path.splitext(os.path.basename(output_path))[1].replace(".", "")
        embedder_model = embedders_dropdown.get()
        f0_autotune = bool(autotune_checkbox.get())
        f0_autotune_strength = round(autotune_strength_slider.get(), 1)
        split_audio = bool(split_audio_checkbox.get())
        clean_audio = bool(clean_audio_checkbox.get())
        clean_strength = round(clean_strength_slider.get(), 1)
        formant_shifting = bool(formant_shifting_checkbox.get())
        formant_qfrency = round(formant_qfrency_slider.get(), 1)
        formant_timbre = round(formant_timbre_slider.get(), 1)
        proposal_pitch = bool(proposal_pitch_checkbox.get())
        proposal_pitch_threshold = round(proposal_pitch_threshold_slider.get(), 1)
    except Exception as e:
        results = "[ERROR] Error occurred while retrieving parameters."
        print(f"[ERROR] An error has occurred: {e}")
    else:
        if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith(".pth"):
            print("[WARNING] Please enter a valid model.")
            results = "[WARNING] Please enter a valid model."
        elif not os.path.exists(input_path):
            print("[WARNING] No audio files found.")
            results = "[WARNING] No audio files found."
        else:
            print(f"[INFO] Model Path: {pth_path} Index Path: {index_path}")
            print(f"[INFO] Pitch: {pitch} Index Rate: {index_rate} RMS Mix Rate: {volume_envelope} Protect: {protect} Hop Length: {hop_length} F0 Method: {f0_method} Embedder Model: {embedder_model} AutoTune: {f0_autotune} AutoTune Strength: {f0_autotune_strength} Split Audio: {split_audio} Clean Audio: {clean_audio} Clean Strength: {clean_strength} Formant Shifting: {formant_shifting} Formant Qfrency: {formant_qfrency} Formant Timbre: {formant_timbre} Proposal Pitch: {proposal_pitch} Proposal Pitch Threshold: {proposal_pitch_threshold}")
            
            process_frame.place(x=100, y=10)
            conversion_button.configure(state=tk.DISABLED)
            loading_progress.start()

            try:
                print(f"[INFO] Run inference on audio file...")

                run_inference_script(
                    config=config,
                    pitch=pitch, 
                    filter_radius=3, 
                    index_rate=index_rate, 
                    volume_envelope=volume_envelope, 
                    protect=protect, 
                    hop_length=hop_length, 
                    f0_method=f0_method, 
                    input_path=input_path, 
                    output_path=output_path, 
                    pth_path=pth_path, 
                    index_path=index_path, 
                    export_format=export_format, 
                    embedder_model=embedder_model, 
                    resample_sr=48000 if export_format.lower() != "wav" else 0,  
                    f0_autotune=f0_autotune, 
                    f0_autotune_strength=f0_autotune_strength, 
                    split_audio=split_audio,
                    clean_audio=clean_audio, 
                    clean_strength=clean_strength,
                    formant_shifting=formant_shifting,
                    formant_qfrency=formant_qfrency, 
                    formant_timbre=formant_timbre,
                    proposal_pitch=proposal_pitch,
                    proposal_pitch_threshold=proposal_pitch_threshold
                )

                if not os.path.exists(output_path) or not os.path.getsize(output_path) > 0:
                    print("[ERROR] It seems an error occurred during inference. No output audio file found.")
                    results = "[ERROR] It seems an error occurred during inference. No output audio file found."
                else:
                    print(f"[INFO] Inference completed, your output is: {output_path}.")
                    results = f"[INFO] Inference completed, your output is: {output_path}."

                    open_output_button.place(relx=0.5, y=10, anchor="n")
                    audio_load = output_path
            except RuntimeError as e:
                results = "[ERROR] An error occurred while running audio inference."
                print(f"[ERROR] An error has occurred: {e}")

    output_path_frame.delete("0.0", tk.END)
    output_path_frame.insert("0.0", process_output(input_path))

    result_state.configure(text_color=("#00D612" if not results.startswith("[WARNING]") else "#FBFF00") if not results.startswith("[ERROR]") else "#FF0000")
    result_state.place(relx=0.5, y=60, anchor="n")
    result_state.configure(text=results)

    conversion_button.configure(state=tk.NORMAL)
    process_frame.place_forget()
    loading_progress.stop()

def start_processing():
    t = threading.Thread(target=on_click)
    t.start()

# Nền giao diện chính

root = ctk.CTk()
root.title("RVC GUI")

root.geometry("800x825")
root.resizable(False, False)
root.configure(fg_color="#262626")

above_left = ctk.CTkFrame(
    master=root,
    width=447,
    height=293,
    corner_radius=15,
    fg_color="#404040"
)

bottom_left = ctk.CTkFrame(
    master=root,
    width=447,
    height=507,
    corner_radius=15,
    fg_color="#404040"
)

# Khung bên phải

right_frame = ctk.CTkFrame(
    master=root,
    width=325,
    height=805,
    corner_radius=15,
    fg_color="#404040"
)

# Đặt vị trí của khung

above_left.place(x=11, y=10)
bottom_left.place(x=11, y=308)
right_frame.place(x=465, y=10)

# Khung trong bên trái

left_above_frame = ctk.CTkFrame(
    master=above_left,
    width=423,
    height=122,
    corner_radius=15,
    fg_color="#262626"
)

left_bottom_frame = ctk.CTkFrame(
    master=above_left,
    width=423,
    height=147,
    corner_radius=15,
    fg_color="#262626"
)

# Khung trong bên phải

right_above_frame = ctk.CTkFrame(
    master=right_frame,
    width=307,
    height=147,
    corner_radius=15,
    fg_color="#262626"
)

right_middle_frame = ctk.CTkFrame(
    master=right_frame,
    width=307,
    height=159,
    corner_radius=15,
    fg_color="#262626"
)

right_bottom_frame = ctk.CTkFrame(
    master=right_frame,
    width=307,
    height=474,
    corner_radius=15,
    fg_color="#262626"
)

# Đặt vị trí của khung

left_above_frame.place(x=12, y=10)
left_bottom_frame.place(x=12, y=136)

right_above_frame.place(x=9, y=8)
right_middle_frame.place(x=9, y=161)
right_bottom_frame.place(x=9, y=326)

# Đầu vào âm thanh

input_audio_label = ctk.CTkLabel(
    master=left_above_frame, 
    width=86,
    height=13,
    text="INPUT PATH"
)

input_path_frame = ctk.CTkTextbox(
    master=left_above_frame,
    width=186,
    height=39,
    corner_radius=10,
    fg_color="#595959",
    border_width=2,
    border_color="#404040"
)

browse_button = ctk.CTkButton(
    master=left_above_frame, 
    text="Select File", 
    width=112,
    height=39,
    corner_radius=10,
    fg_color="#4e95d9",
    command=browse_file
)

# Đầu ra âm thanh

output_audio_label = ctk.CTkLabel(
    master=left_above_frame, 
    width=86,
    height=13,
    text="OUTPUT PATH"
)

output_path_frame = ctk.CTkTextbox(
    master=left_above_frame,
    width=304,
    height=39,
    corner_radius=10,
    fg_color="#595959",
    border_width=2,
    border_color="#404040"
)

# Vị trí đầu vào, ra âm thanh và chọn tệp

input_audio_label.place(x=10, y=21)
input_path_frame.place(x=100, y=10)

browse_button.place(x=290, y=10)

output_audio_label.place(x=10, y=81)
output_path_frame.place(x=100, y=70)

# Phần trích xuất nhúng và trích xuất cao độ

f0_method_label = ctk.CTkLabel(
    master=left_bottom_frame, 
    width=48,
    height=13,
    text="F0 Method",
)

f0_method_dropdown = ctk.CTkOptionMenu(
    master=left_bottom_frame,
    width=129,
    height=39,
    corner_radius=10,
    fg_color="#595959",
    values=f0_method_full_list,
    dynamic_resizing=False,
    command=lambda _: refresh_slider_positions()
)

embedders_label = ctk.CTkLabel(
    master=left_bottom_frame, 
    width=48,
    height=13,
    text="EMBEDDER"
)

embedders_dropdown = ctk.CTkOptionMenu(
    master=left_bottom_frame,
    width=129,
    height=39,
    corner_radius=10,
    fg_color="#595959",
    values=embedders_full_list,
    dynamic_resizing=False
)

# Vị trí trích xuất cao độ, nhúng và đặt mặc định

f0_method_label.place(x=10, y=21)
f0_method_dropdown.place(x=75, y=10)

embedders_label.place(x=215, y=21)
embedders_dropdown.place(x=285, y=10)

f0_method_dropdown.set("rmvpe")
embedders_dropdown.set("hubert_base")

# Thanh trượt cao độ

pitch_label = ctk.CTkLabel(
    master=left_bottom_frame, 
    width=86,
    height=13,
    text="PITCH: 0"
)

pitch_slider = ctk.CTkSlider(
    master=left_bottom_frame,
    width=300,
    height=18,
    corner_radius=10,
    from_=-20,         
    to=20, 
    number_of_steps=100,
    command=pitch_slider_event
)

# Vị trí của thanh trượt cao độ và đặt mặc định

pitch_label.place(x=320, y=70)
pitch_slider.place(x=10, y=70)

pitch_slider.set(0)

# Nút chuyển đổi

conversion_button = ctk.CTkButton(
    master=left_bottom_frame, 
    width=160,
    height=37,
    corner_radius=15,
    fg_color="#4ea72e", 
    text="Conversion", 
    command=start_processing
)

# Vị trí nút chuyển đổi

conversion_button.place(x=131, y=100)

# Thanh quá trình và kết quả

process_frame = ctk.CTkFrame(
    master=bottom_left,
    width=247,
    height=60,
    corner_radius=15,
    fg_color="#262626"
)

loading_label = ctk.CTkLabel(
    master=process_frame, 
    width=147,
    height=18,
    text="Converting. Please Wait..."
)

loading_progress = ctk.CTkProgressBar(
    master=process_frame, 
    width=147,
    height=5,
    mode="indeterminate"
)

result_state = ctk.CTkLabel(
    master=bottom_left, 
    text="", 
    width=200, 
    height=50, 
    corner_radius=10,
    text_color="#00D612",
    wraplength=300
)

open_output_button = ctk.CTkButton(
    master=bottom_left, 
    text="Open", 
    width=100,
    height=40,
    corner_radius=10,
    fg_color="#2471b9",
    command=play_audio
)

# Vị trí thanh quá trình và kết quả

#result_state.place(relx=0.5, y=60, anchor="n")
#process_frame.place(x=100, y=10)
#open_output_button.place(relx=0.5, y=10, anchor="n")

loading_label.place(x=50, y=10)
loading_progress.place(x=50, y=40)

# Phần mẹo thông tin

notes_label = ctk.CTkLabel(
    master=bottom_left, 
    justify="left", 
    text_color="#8A8A8A", 
    text="""
    Tips:
    1. Mangio Crepe is the crepe model for Applio.
    2. Crepe has additional mean and median filters.
    3. Yin and pYin are fast but not of high quality.
    4. Pm is fast but the output quality is not high.
    5. Dio would probably be a good fit for rap music.
    6. Swipe is written in numpy so it will run on cpu.
    7. Rmvpe provides good quality, fast and is the best choice right now.
    8. Fcpe takes less context so it will be fast but may lose information.
    9. Harvest provides high quality but is slow and often has errors at the begin.
    """
)

# Vị trí mẹo thông tin

notes_label.place(x=1, y=325)

# Tải lên mô hình

import_models_button = ctk.CTkButton(
    master=right_above_frame, 
    width=186,
    height=39,
    corner_radius=15,
    fg_color="#ff0000", 
    hover_color="#c50000",
    text="Import Model (.zip)", 
    command=browse_zip
)

# Vị trí nút tải lên

import_models_button.place(x=60, y=10)

# Phần chọn bộ xử lí

processing_device_label = ctk.CTkLabel(
    master=right_above_frame, 
    width=186,
    height=39,
    text="Processing Device"
)

change_device = ctk.CTkSegmentedButton(
    master=right_above_frame, 
    width=142,
    height=43,
    values=["GPU", "CPU"],
    command=lambda value: update_config(value)
)

# Vị trí và đặt mặc định phần xử lí

if not config.device.startswith("cpu"): change_device.set("GPU")
else:
    change_device.configure(state=tk.DISABLED)
    change_device.set("CPU")

change_device.place(x=110, y=85)
processing_device_label.place(x=60, y=50)

# Mô hình và chỉ mục

select_model = ctk.StringVar(
    value="Select a model"
)

model_path_label = ctk.CTkLabel(
    master=right_middle_frame, 
    width=13,
    height=7,
    text="MODEL PATH"
)

model_path_dropdown = ctk.CTkOptionMenu(
    master=right_middle_frame,
    width=291,
    height=39,
    corner_radius=10,
    fg_color="#595959",
    values=model_folders,
    variable=select_model,
    command=index_select
)

index_path_label = ctk.CTkLabel(
    master=right_middle_frame, 
    width=13,
    height=7,
    text="INDEX PATH (Recommended)"
)

index_path_textbox = ctk.CTkTextbox(
    master=right_middle_frame,
    width=291,
    height=39,
    corner_radius=10,
    fg_color="#595959",
    border_width=2,
    border_color="#404040"
)

# Vị trí của chọn mô hình và chỉ mục

model_path_label.place(x=15, y=15)
model_path_dropdown.place(x=10, y=35)

index_path_label.place(x=15, y=85)
index_path_textbox.place(x=10, y=105)

# Phần cài đặt checkbox

is_half_checkbox = ctk.CTkCheckBox(
    master=right_bottom_frame,
    width=5,
    height=5,
    corner_radius=10,
    fg_color="#00629B",
    border_width=2,
    border_color="#404040",
    text="IS HALF",
    state=tk.NORMAL if config.device.startswith("cuda") else tk.DISABLED,
    command=update_is_half
)

clean_audio_checkbox = ctk.CTkCheckBox(
    master=right_bottom_frame,
    width=5,
    height=5,
    corner_radius=10,
    fg_color="#00629B",
    border_width=2,
    border_color="#404040",
    text="Noise Reduce",
    command=refresh_slider_positions
)

autotune_checkbox = ctk.CTkCheckBox(
    master=right_bottom_frame,
    width=5,
    height=5,
    corner_radius=10,
    fg_color="#00629B",
    border_width=2,
    border_color="#404040",
    text="AutoTune",
    command=refresh_slider_positions
)

split_audio_checkbox = ctk.CTkCheckBox(
    master=right_bottom_frame,
    width=5,
    height=5,
    corner_radius=10,
    fg_color="#00629B",
    border_width=2,
    border_color="#404040",
    text="Split Audio",
)

formant_shifting_checkbox = ctk.CTkCheckBox(
    master=right_bottom_frame,
    width=5,
    height=5,
    corner_radius=10,
    fg_color="#00629B",
    border_width=2,
    border_color="#404040",
    text="Formant Shift",
    command=refresh_slider_positions
)

proposal_pitch_checkbox = ctk.CTkCheckBox(
    master=right_bottom_frame,
    width=5,
    height=5,
    corner_radius=10,
    fg_color="#00629B",
    border_width=2,
    border_color="#404040",
    text="Proposal Pitch",
    command=refresh_slider_positions
)

# Vị trí các checkbox

is_half_checkbox.place(x=10, y=10)
clean_audio_checkbox.place(x=160, y=10)

autotune_checkbox.place(x=10, y=40)
split_audio_checkbox.place(x=160, y=40)

formant_shifting_checkbox.place(x=10, y=70)
proposal_pitch_checkbox.place(x=160, y=70)

# Phần cài đặt thanh trượt

index_strength_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Index Strength: 0.5"
)

index_strength_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=1, 
    number_of_steps=100,
    command=index_strength_event
)

protect_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Protecting: 0.33"
)

protect_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=1, 
    number_of_steps=100,
    command=protect_event
)

rms_mix_rate_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="RMS Mix Rate: 1"
)

rms_mix_rate_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=1, 
    number_of_steps=100,
    command=rms_mix_rate_event
)

hop_length_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Hop Length: 128"
)

hop_length_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=1,         
    to=8, 
    number_of_steps=7,
    command=hop_length_event
)

clean_strength_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Clean Strength: 0.5"
)

clean_strength_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=1, 
    number_of_steps=100,
    command=clean_strength_event
)

autotune_strength_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="AutoTune Strength: 1"
)

autotune_strength_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=1, 
    number_of_steps=100,
    command=autotune_strength_event
)

formant_qfrency_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Formant Qfrency: 1.0"
)

formant_qfrency_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=16, 
    number_of_steps=100,
    command=formant_qfrency_event
)

formant_timbre_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Formant Timbre: 1.0"
)

formant_timbre_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=0,         
    to=16, 
    number_of_steps=100,
    command=formant_timbre_event
)

proposal_pitch_threshold_label = ctk.CTkLabel(
    master=right_bottom_frame, 
    width=111,
    height=22,
    text="Proposal Pitch: 155.0"
)

proposal_pitch_threshold_slider = ctk.CTkSlider(
    master=right_bottom_frame,
    width=281,
    height=18,
    corner_radius=10,
    from_=50,         
    to=1200, 
    number_of_steps=1150,
    command=proposal_pitch_threshold_event
)

# Vị trí phần thanh trượt và đặt mặc định

index_strength_label.place(x=95, y=100)
index_strength_slider.place(x=15, y=120)

protect_label.place(x=95, y=140)
protect_slider.place(x=15, y=160)

rms_mix_rate_label.place(x=95, y=180)
rms_mix_rate_slider.place(x=15, y=200)

# hop_length_label.place(x=95, y=220)
# hop_length_slider.place(x=15, y=240)

# clean_strength_label.place(x=95, y=260)
# clean_strength_slider.place(x=15, y=280)

# autotune_strength_label.place(x=95, y=300)
# autotune_strength_slider.place(x=15, y=320)

# formant_qfrency_label.place(x=95, y=340)
# formant_qfrency_slider.place(x=15, y=360)

# formant_timbre_label.place(x=95, y=380)
# formant_timbre_slider.place(x=15, y=400)

# proposal_pitch_threshold_label.place(x=95, y=420)
# proposal_pitch_threshold_slider.place(x=15, y=440)

index_strength_slider.set(0.5)
protect_slider.set(0.33)

rms_mix_rate_slider.set(1)
hop_length_slider.set(2)

clean_strength_slider.set(0.5)
autotune_strength_slider.set(1)

formant_qfrency_slider.set(1)
formant_timbre_slider.set(1)

proposal_pitch_threshold_slider.set(155.0)

root.mainloop()
