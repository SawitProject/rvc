#@title **🖥️ Ipywidgets UI**
import os
import torch
import threading

from IPython.display import display
from ipywidgets import HBox, VBox, Text, Label, Dropdown, FileUpload, Layout, IntSlider, FloatSlider, Button, HTML, Checkbox

cpu_mode = False
is_half = False

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

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

model_folders = sorted(
    [
        os.path.join(
            root,
            name
        )
        for root, _, files in os.walk(
            "rvc_models",
            topdown=False
        )
        for name in files
        if name.endswith(".pth")
    ]
)

def process_output(file_path):
    if not os.path.exists(file_path): return file_path
    file = os.path.splitext(os.path.basename(file_path))

    index = 1
    while 1:
        file_path = os.path.join(os.path.dirname(file_path), f"{file[0]}_RVC_{index}{file[1]}")
        if not os.path.exists(file_path): return file_path
        index += 1

def handle_upload(_):
    try:
        for file_info in select_file.value:
            name = file_info["name"]
            content = file_info["content"]

            with open(name, "wb") as f:
                f.write(content)
    except:
        if select_file.value:
            uploaded = list(select_file.value.values())[0]
            name = list(select_file.value.keys())[0]
            content = uploaded['content']

            with open(name, 'wb') as f:
                f.write(content)

    input_path.value = name
    output_path.value = process_output(name)

def update_model_value(_):
    global model_folders

    model_folders = sorted(
        [
            os.path.join(
                root,
                name
            )
            for root, _, files in os.walk(
                "rvc_models",
                topdown=False
            )
            for name in files 
            if name.endswith(".pth")
        ]
    )

    model_path.options = model_folders

def index_select(choice):
    dirs = os.path.dirname(choice["new"])

    for f in os.listdir(dirs):
        if f.endswith(".index") and "trained" not in f:
            index_path.value = os.path.join(dirs, f)
            return

    index_path.value = ""

def refresh_slider_positions(_):
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

    f0_value = f0_method.value

    if f0_value in show_hop_length_method:
        hop_length_box.layout.display = "flex"
    else:
        hop_length_box.layout.display = "none"

    if noise_reduce.value:
        clean_strength_box.layout.display = "flex"
    else:
        clean_strength_box.layout.display = "none"

    if autotune.value:
        autotune_box.layout.display = "flex"
    else:
        autotune_box.layout.display = "none"

def update_is_half(_):
    global cpu_mode, is_half

    is_half = bool(is_half_checkbox.value) and torch.cuda.is_available() and not cpu_mode

def on_click(_):
    centered_result_state.layout.display = "none"
    centered_loading_label.layout.display = "none"

    try:
        _pitch = pitch.value
        index_rate = index_strength.value
        volume_envelope = rms_mix_rate.value
        _protect = protect.value
        _hop_length = hop_length.value
        _f0_method = f0_method.value
        _input_path = (input_path.value).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        _output_path = (output_path.value or process_output(_input_path)).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        pth_path = (model_path.value).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        _index_path = (index_path.value).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        export_format = os.path.splitext(os.path.basename(_output_path))[1].replace(".", "")
        embedder_model = embedders.value
        f0_autotune = bool(autotune.value)
        f0_autotune_strength = autotune_slider.value
        _split_audio = bool(split_audio.value)
        clean_audio = bool(noise_reduce.value)
        _clean_strength = clean_strength.value
    except Exception as e:
        results = "[ERROR] Error occurred while retrieving parameters."
        print(f"[ERROR] An error has occurred: {e}")
    else:
        if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith(".pth"):
            print("[WARNING] Please enter a valid model.")
            results = "[WARNING] Please enter a valid model."
        elif not os.path.exists(_input_path):
            print("[WARNING] No audio files found.")
            results = "[WARNING] No audio files found."
        else:
            print(f"[INFO] Model Path: {pth_path} Index Path: {_index_path}")
            print(f"[INFO] Pitch: {_pitch} Index Rate: {index_rate} RMS Mix Rate: {volume_envelope} Protect: {_protect} Hop Length: {_hop_length} F0 Method: {_f0_method} Embedder Model: {embedder_model} AutoTune: {f0_autotune} AutoTune Strength: {f0_autotune_strength} Split Audio: {_split_audio} Clean Audio: {clean_audio} Clean Strength: {_clean_strength}")

            conversion_button.disabled = True
            centered_loading_label.layout.display = "flex"

            try:
                print(f"[INFO] Run inference on audio file...")
                resample_sr = 48000 if export_format.lower() != "wav" else 0

                args = f'from modules.inference import run_inference_script; run_inference_script(is_half={is_half}, cpu_mode={cpu_mode}, pitch={_pitch}, filter_radius=3, index_rate={index_rate}, volume_envelope={volume_envelope}, protect={_protect}, hop_length={_hop_length}, f0_method=\\"{_f0_method}\\", input_path=\\"{_input_path}\\", output_path=\\"{_output_path}\\", pth_path=\\"{pth_path}\\", index_path=\\"{_index_path}\\", export_format=\\"{export_format}\\", embedder_model=\\"{embedder_model}\\", resample_sr={resample_sr}, f0_autotune={f0_autotune}, f0_autotune_strength={f0_autotune_strength}, split_audio={_split_audio}, clean_audio={clean_audio}, clean_strength={_clean_strength})'
                !python3 -c "$args"

                if not os.path.exists(_output_path) or not os.path.getsize(_output_path) > 0:
                    print("[ERROR] It seems an error occurred during inference. No output audio file found.")
                    results = "[ERROR] It seems an error occurred during inference. No output audio file found."
                else:
                    print(f"[INFO] Inference completed, your output is: {_output_path}.")
                    results = f"[INFO] Inference completed, your output is: {_output_path}."
            except RuntimeError as e:
                results = "[ERROR] An error occurred while running audio inference."
                print(f"[ERROR] An error has occurred: {e}")

    output_path.value = process_output(_input_path)

    centered_result_state.layout.display = "flex"
    centered_loading_label.layout.display = "none"

    result_state.style.color = ("#00D612" if not results.startswith("[WARNING]") else "#FBFF00") if not results.startswith("[ERROR]") else "#FF0000"
    result_state.value = results

    conversion_button.disabled = False

# def start_processing(_):
#     t = threading.Thread(target=on_click)
#     t.start()

input_path = Text(
    description="INPUT PATH:",
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto",
        min_width="390px"
    )
)

select_file = FileUpload(
    description="Select File",
    accept=".wav, .mp3, .flac, .ogg, .opus, .m4a, .mp4, .aac, .alac, .wma, .aiff, .webm, .ac3",
    multiple=False,
    style={
        "description_width": "initial"
    },
    button_style="primary",
    layout=Layout(
        width="auto"
    )
)

input_frame = HBox(
    [
        input_path,
        select_file
    ],
    layout=Layout(
        gap="10px"
    )
)

output_path = Text(
    description="OUTPUT PATH:",
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto"
    )
)

select_file.observe(handle_upload, names="value")

left_above_frame = VBox(
    [
        input_frame,
        output_path
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="530px",
        height="100px"
    )
)

f0_method = Dropdown(
    options=f0_method_full_list,
    value="rmvpe",
    description="F0 Method",
    style={
        "description_width": "initial"
    }
)

embedders = Dropdown(
    options=embedders_full_list,
    value="hubert_base",
    description="Embedder",
    style={
        "description_width": "initial"
    }
)

extract = HBox(
    [
        f0_method,
        embedders
    ],
    layout=Layout(
        justify_content="center"
    )
)

f0_method.observe(refresh_slider_positions, names="value")

pitch = IntSlider(
    value=0,
    min=-20,
    max=20,
    step=1,
    description="Pitch",
    style={"description_width": "initial"},
    layout=Layout(
        width="500px",
        height="45px"
    )
)

conversion_button = Button(
    description="Conversion",
    button_style="success",
    layout=Layout(
        width="200px",
        height="40px"
    )
)

centered_button = HBox(
    [
        conversion_button
    ],
    layout=Layout(
        justify_content="center"
    )
)

conversion_button.on_click(on_click)

left_middle_frame = VBox(
    [
        extract,
        pitch,
        centered_button
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="530px",
        height="200px"
    )
)

loading_label = Label(
    value="Converting. Please Wait...",
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto",
        height="150px"
    )
)

centered_loading_label = HBox(
    [
        loading_label
    ],
    layout=Layout(
        justify_content="center",
        display="none"
    )
)

result_state = Label(
    value="",
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto",
        height="150px"
    )
)

centered_result_state = HBox(
    [
        result_state
    ],
    layout=Layout(
        justify_content="center",
        display="none"
    )
)

tips_html = """
<div style="
    font-family: Arial, sans-serif;
    font-size: 14px;
    color: #aaa;
    background-color: transparent;
    padding: 10px;
    line-height: 1.6;
">
<b style="color: #ccc;">Tips:</b><br>
1. Mangio Crepe is the crepe model for Applio.<br>
2. Crepe has additional mean and median filters.<br>
3. Yin and pYin are fast but not of high quality.<br>
4. Pm is fast but the output quality is not high.<br>
5. Dio would probably be a good fit for rap music.<br>
6. Swipe is written in numpy so it will run on cpu.<br>
7. Rmvpe provides good quality, fast and is the best choice right now.<br>
8. Fcpe takes less context so it will be fast but may lose information.<br>
9. Harvest provides high quality but is slow and often has errors at the begin.
</div>
"""

tip_label = HTML(
    value=tips_html
)

left_bottom_frame = VBox(
    [
        centered_loading_label,
        centered_result_state,
        tip_label
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="530px",
        height="500px",
        justify_content="flex-end"
    )
)

left_frame = VBox(
    [
        left_above_frame,
        left_middle_frame,
        left_bottom_frame
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="552px",
        height="750px"
    )
)

refresh_button = Button(
    description="Refresh",
    button_style="danger",
    layout=Layout(
        width="150px",
        height="40px"
    )
)

centered_button_refresh = HBox(
    [
        refresh_button
    ],
    layout=Layout(
        justify_content="center"
    )
)

right_above_frame = VBox(
    [
        centered_button_refresh
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="378px",
        height="70px"
    )
)

model_label = Label(
    value="Model Path",
    layout=Layout(
        width="auto"
    )
)

model_path = Dropdown(
    layout=Layout(
        width="auto"
    ),
    options=model_folders,
    value=None
)

model_box = VBox(
    [
        model_label,
        model_path
    ]
)

index_label = Label(
    value="Index Path",
    layout=Layout(
        width="auto"
    )
)

index_path = Text(
    layout=Layout(
        width="auto"
    )
)

index_box = VBox(
    [
        index_label,
        index_path
    ]
)

refresh_button.on_click(update_model_value)
model_path.observe(index_select, names="value")

right_middle_frame = VBox(
    [
        model_box,
        index_box
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="378px",
        height="150px"
    )
)

is_half_checkbox = Checkbox(
    value=False,
    description="IS HALF",
    indent=False,
    disabled=not torch.cuda.is_available()
)

noise_reduce = Checkbox(
    value=False,
    description="Noise Reduce",
    indent=False
)

top_row = HBox(
    [
        is_half_checkbox,
        noise_reduce
    ],
    layout=Layout(
        gap="20px"
    )
)

autotune = Checkbox(
    value=False,
    description="AutoTune",
    indent=False
)

split_audio = Checkbox(
    value=False,
    description="Split Audio",
    indent=False
)

bottom_row = HBox(
    [
        autotune,
        split_audio
    ],
    layout=Layout(
        gap="20px"
    )
)

index_label = Label(
    value="Index Strength",
    layout=Layout(
        width="auto"
    )
)

index_strength = FloatSlider(
    value=0.5,
    min=0,
    max=1,
    step=0.01,
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto"
    )
)

index_strength_box = VBox(
    [
        index_label,
        index_strength
    ],
)

protect_label = Label(
    value="Protecting",
    layout=Layout(
        width="auto"
    )
)

protect = FloatSlider(
    value=0.33,
    min=0,
    max=1,
    step=0.01,
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto"
    )
)

protect_box = VBox(
    [
        protect_label,
        protect
    ],
)

rms_mix_rate_label = Label(
    value="RMS Mix Rate",
    layout=Layout(
        width="auto"
    )
)

rms_mix_rate = FloatSlider(
    value=1,
    min=0,
    max=1,
    step=0.1,
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto"
    )
)

rms_mix_rate_box = VBox(
    [
        rms_mix_rate_label,
        rms_mix_rate
    ],
)

hop_length_label = Label(
    value="Hop Length",
    layout=Layout(
        width="auto"
    )
)

hop_length = IntSlider(
    value=128,
    min=64,
    max=512,
    step=32,
    style={
        "description_width": "initial"
    },
    layout=Layout(width="auto")
)

hop_length_box = VBox(
    [
        hop_length_label,
        hop_length
    ],
    layout=Layout(display="none")
)

clean_strength_label = Label(
    value="Clean Strength",
    layout=Layout(
        width="auto"
    )
)

clean_strength = FloatSlider(
    value=0.5,
    min=0,
    max=1,
    step=0.1,
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto"
    )
)

clean_strength_box = VBox(
    [
        clean_strength_label,
        clean_strength
    ],
    layout=Layout(display="none")
)

autotune_label = Label(
    value="AutoTune Strength",
    layout=Layout(
        width="auto"
    )
)

autotune_slider = FloatSlider(
    value=1,
    min=0,
    max=1,
    step=0.1,
    style={
        "description_width": "initial"
    },
    layout=Layout(
        width="auto"
    )
)

autotune_box = VBox(
    [
        autotune_label,
        autotune_slider
    ],
    layout=Layout(display="none")
)

is_half_checkbox.observe(update_is_half, names="value")
noise_reduce.observe(refresh_slider_positions, names="value")
autotune.observe(refresh_slider_positions, names="value")

right_bottom_frame = VBox(
    [
        top_row,
        bottom_row,
        index_strength_box,
        protect_box,
        rms_mix_rate_box,
        hop_length_box,
        clean_strength_box,
        autotune_box
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="378px",
        height="500px",
    )
)

right_frame = VBox(
    [
        right_above_frame,
        right_middle_frame,
        right_bottom_frame
    ],
    layout=Layout(
        border="2px solid gray",
        padding="10px",
        border_radius="8px",
        width="400px",
        height="750px"
    )
)

gui = HBox(
    [
        left_frame,
        right_frame
    ],
    layout=Layout(
        gap="20px"
    )
)

display(gui)
