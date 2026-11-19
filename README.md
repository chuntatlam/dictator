# Local Push-to-Talk Dictation for macOS

Hold **fn**, speak, release. Your speech is transcribed and cleaned up (punctuation, filler words, self-corrections) by local models, with your own wording kept, then pasted at your cursor in whatever app you're using. Everything runs on-device on Apple Silicon. Audio and text never leave your Mac.

A local, DIY alternative to tools like Wispr Flow.

## Features

- **Push-to-talk with fn**: hold to record, release to transcribe and paste. fn combos (fn+arrows, fn+F-keys) keep working as normal.
- **Fully local and offline**: Parakeet for speech-to-text and Qwen3 for cleanup, both running with [MLX](https://github.com/ml-explore/mlx).
- **Your words, verbatim**: the cleanup only fixes punctuation and casing and removes fillers, stutters, and explicit self-corrections. It never rephrases, swaps synonyms, or fixes grammar.
- **Selectable tone**: `formal` or `casual` formatting, switchable live.
- **Custom dictionary**: names, jargon, and forced replacements, hot-reloaded.
- **Swappable models**: change the ASR or cleanup model with one line or an environment variable.
- **Docker-style pull policy**: models are downloaded only when missing; otherwise the script makes no network requests.
- **Auto-exit**: stops itself after a period without dictation, freeing all model memory.

## How it works

```
hold fn -> record mic -> release fn
        -> ASR model (Parakeet) transcribes audio
        -> dictionary replacements applied
        -> LLM (Qwen3) formats the text in the chosen tone, keeping your wording
        -> dictionary replacements applied again
        -> text is pasted at the cursor (Cmd+V)
```

- **ASR:** [Parakeet TDT 0.6B v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) (English) via [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx)
- **Cleanup LLM:** Qwen3-4B-Instruct (4-bit) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm)
- **Hotkey:** a listen-only macOS event tap on the fn key (PyObjC / Quartz)
- **Lifecycle:** you start it manually in a terminal. It loads and warms the models once, stays resident, and exits after the idle timeout (default 30 min) or on Ctrl+C.

## Requirements

- A Mac with **Apple Silicon** (M1 or newer). MLX does not run on Intel Macs.
- **macOS 13+**
- **Python 3.12+** (developed on 3.14, also tested on 3.12)
- **ffmpeg**
- About **5 GB of free RAM** at peak with the default models (about 3.5-4 GB while idle between dictations; very long recordings may push the peak a little higher), and about **4.5 GB of disk** for the model downloads. The defaults are:
  - **Speech-to-text:** Parakeet TDT 0.6B v2 (`mlx-community/parakeet-tdt-0.6b-v2`), 2.3 GB download
  - **Cleanup LLM:** Qwen3-4B-Instruct-2507, 4-bit (`mlx-community/Qwen3-4B-Instruct-2507-4bit`), 2.1 GB download
  - Other models need different amounts of RAM. See "Changing the models".
- Internet access **only when a model needs to be downloaded** (the first run, or when you switch models). After that the script runs fully offline.

Developed on a MacBook Pro M3 Pro (18 GB).

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/chuntatlam/dictator.git
cd dictator
```

### 2. Install ffmpeg

```bash
brew install ffmpeg
```

(No Homebrew? See [brew.sh](https://brew.sh).)

### 3. Create a virtual environment and install packages

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

`--python 3.14` makes uv use (and download if needed) the Python version the script was developed on. Without it, uv may pick an older Python on your system that the pinned packages don't support.

Or with plain `venv` and `pip` (your `python3` must be 3.12 or newer; macOS's built-in `python3` is too old):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the script was developed against. To install the latest versions instead (unpinned, may break if a package API changes):

```bash
pip install parakeet-mlx mlx-lm soundfile sounddevice numpy pyobjc-framework-Quartz
```

Check your Python version with `python --version`. It must be 3.12 or newer: several pinned packages (`numpy`, `scipy`, `librosa`) require it.

### 4. Change the Globe/fn key setting

Otherwise a press of the key may also trigger the emoji picker or system dictation.

**System Settings -> Keyboard -> "Press 🌐 key to" -> Do Nothing**

(On some keyboards this is labeled "Press fn key to".) This is a system-wide setting. Set it back if you stop using this tool and want the old behavior.

### 5. Grant permissions

macOS requires permissions for whichever app you run the script from (Terminal, iTerm, etc.). Open **System Settings -> Privacy & Security** and enable your terminal app under:

| Permission | Why |
|---|---|
| **Input Monitoring** | Detect fn key presses |
| **Accessibility** | Send the Cmd+V paste keystroke |
| **Microphone** | Record audio (the prompt appears on your first fn press) |

Input Monitoring and Accessibility often don't prompt automatically. Add your terminal app manually with the `+` button if needed, then restart the terminal.

Permissions are tied to the app you launch from. If you switch terminal apps, grant them again for the new one.

## Running

```bash
source .venv/bin/activate
python dictate.py
```

**First run:** the script checks for the models in your local cache and **downloads any that are missing** (about 4.5 GB with the default models, a few minutes). Make sure you're online for this run. Progress bars show in the terminal. You may see Hugging Face warnings about unauthenticated requests; they're harmless, and no token is needed for these public models.

**Later runs:** the models load from the cache with no network access.

Wait for the `ready in X.Xs` line. Nothing responds to fn until it appears. Then:

1. Click into any text field in any app.
2. **Hold fn**, speak, **release fn**. The cleaned text is pasted at the cursor after a moment.
3. Stop with **Ctrl+C**, or let it exit after the idle timeout.

The orange microphone indicator in the menu bar is only on while fn is held. The mic is closed at all other times.

Optional shell alias:

```bash
alias dictate='cd ~/path/to/<your-repo> && source .venv/bin/activate && python dictate.py'
```

### Behavior notes

- **Clipboard:** text is inserted with `pbcopy` + Cmd+V, so it appears instantly but **overwrites your clipboard** with the dictated text.
- **Accidental taps:** recordings shorter than ~0.3 s are ignored.
- **fn as a modifier:** if you press another key while holding fn (e.g. fn+arrow), the audio recorded up to that key press is discarded. Recording continues while fn stays held, so if you keep holding fn for more than ~0.3 s after the last key press, that audio is still transcribed and pasted. Release fn promptly after a fn combo.
- **Fallback:** if the LLM output looks off-task or truncated, the raw transcript is pasted instead.
- **Secure input fields** (e.g. password boxes) block synthetic keystrokes, so pasting will not work there.

## Project layout

`tone.txt` and `dictionary.txt` go in the **project root, in the same folder as `dictate.py`**. The script finds them relative to its own location, so it doesn't matter which directory you launch it from.

```text
<your-repo>/
├── dictate.py         # the script
├── tone.txt           # optional: one word, formal | casual
├── dictionary.txt     # optional: custom words, one entry per line
├── requirements.txt   # pinned package versions
├── README.md
└── .venv/             # your virtual environment
```

Both files are optional and re-read automatically, so you can edit them while the script is running. Use `TONE_FILE=/path/to/tone.txt` or `DICTIONARY=/path/to/dictionary.txt` if you'd rather keep them elsewhere.

## Tone

The tone controls **formatting only**. In every tone your wording is kept exactly as spoken: no rephrasing, no synonyms, no grammar fixes, and contractions and slang stay as you said them.

| Tone | Effect |
|---|---|
| `formal` | Full punctuation, standard capitalization, paragraph breaks for long text |
| `casual` (default) | Light punctuation, relaxed capitalization, no trailing period on short messages |

Set the tone in either of two ways:

- **Live switching (no restart):** write the tone name into `tone.txt` in the project root. It is re-read on every dictation.
  ```bash
  echo formal > tone.txt
  echo casual > tone.txt
  ```
- **Startup default:** `TONE=formal python dictate.py`. Used when `tone.txt` is missing or invalid.

To add or edit tones, change the `TONES` dictionary in `dictate.py`. Keep tone instructions about formatting: an instruction that asks for different wording conflicts with the cleanup prompt. If a small model still rewords your text, try a larger LLM (see "Changing the models").

## Custom dictionary

Add names, jargon, and product terms in `dictionary.txt` in the project root. The file is re-read whenever it changes, so there is no need to restart. Format, one entry per line:

```text
# comments start with a hash
Kubernetes
Keycloak
OpenTelemetry
cube control => kubectl
open fga => OpenFGA
```

- **`Word`**: the LLM is told to spell it exactly this way, and single words (4+ characters) are also fuzzy-matched against the transcript, so close misspellings such as "Keycloack" are corrected.
- **`heard => Correct`**: a deterministic, case-insensitive replacement for things the ASR model consistently gets wrong. Use this for phrases and sound-alikes. The `Correct` side is also treated as a `Word` entry (sent to the LLM and fuzzy-matched).

Replacements and fuzzy matching run both before and after the LLM. Only the first 200 terms are sent to the LLM (`MAX_DICT_TERMS_IN_PROMPT` in `dictate.py`); replacements and fuzzy matching use every entry.

Parakeet does not accept a vocabulary prompt, so the dictionary is applied after transcription (replacements, fuzzy matching, and the LLM prompt), not inside the ASR model. If you switch to Whisper, you can also bias recognition directly by passing the terms as `initial_prompt` to `mlx_whisper.transcribe(...)`.

## Changing the models

The models are set in the config block at the top of `dictate.py`:

```python
ASR_REPO = os.environ.get("ASR", "mlx-community/parakeet-tdt-0.6b-v2")
LLM_REPO = os.environ.get("LLM", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
```

You can change them in two ways:

**Option A: edit the default string** in `dictate.py`.

**Option B: override per run** with an environment variable, without touching the file:

```bash
LLM=mlx-community/Qwen3-8B-4bit python dictate.py
```

A model you haven't used before is **downloaded automatically on first use** (see "Pull policy" below) and cached in `~/.cache/huggingface`. If a repo name is wrong, the download fails and the script exits with an error naming the model.

### Changing the cleanup LLM

Any `mlx-lm`-compatible repo from [mlx-community](https://huggingface.co/mlx-community) should work (Qwen, Llama, Gemma, etc.) by changing `LLM_REPO`.

| Model | Approx. RAM | Notes |
|---|---|---|
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.5 GB | Default. Fast, good enough for most cleanup. |
| `mlx-community/Qwen3-8B-4bit` | ~5 GB | Roughly 2x slower per cleanup, better at preserving meaning. |
| `mlx-community/Qwen3-8B-8bit` | ~9 GB | Higher quality, needs more memory headroom. |

Notes:
- Hybrid "thinking" models such as `Qwen3-8B` are handled: the script passes `enable_thinking=False` and strips any `<think>` blocks. Other reasoning models may need their own handling.
- The cleanup prompt (`SYSTEM`), the example pairs (`FEWSHOT`), and the sanity-check thresholds in `clean()` were tuned loosely for Qwen. A different model family may need prompt tweaks.
- Exact repo names can change. If a download fails, check the current name on the [mlx-community page](https://huggingface.co/mlx-community).

### Changing the ASR model

**Other Parakeet repos** (for example a newer version, or the multilingual v3) work by changing `ASR_REPO`, since the script loads them with `parakeet_mlx.from_pretrained`:

```bash
ASR=mlx-community/parakeet-tdt-0.6b-v3 python dictate.py
```

**Whisper** (broader language coverage) is a different model family, so it needs a small code change:

```bash
pip install mlx-whisper     # or: uv pip install mlx-whisper
```

In `dictate.py`:

```python
import mlx_whisper

# in worker(): delete the line  asr = from_pretrained(ASR_REPO)

def transcribe(audio):
    return mlx_whisper.transcribe(audio, path_or_hf_repo=ASR_REPO)["text"].strip()
```

Then set `ASR_REPO` to a Whisper repo, e.g. `mlx-community/whisper-large-v3-turbo`.

## Pull policy (offline behavior)

The script behaves like Docker's `pull_policy`. The main process is **always offline**: it never contacts Hugging Face itself. At startup it checks the local cache for each model (a purely local check). If a model is missing, it is downloaded by a separate subprocess, then loaded from the cache.

Control this with the `PULL` environment variable:

| `PULL` | Behavior |
|---|---|
| `missing` (default) | Download a model only if it isn't cached. Otherwise no network access. |
| `never` | Never download. Exit with a clear message if a model is missing. |
| `always` | Check Hugging Face for updates on every launch. |

The value is case-insensitive. Any other value behaves like `never`.

```bash
python dictate.py                 # default: pull only what's missing
PULL=never python dictate.py      # strictly offline
PULL=always python dictate.py     # refresh models every launch
```

To verify it's running offline:

1. Turn off Wi-Fi (and unplug Ethernet), then run `python dictate.py`. With the models cached, it should reach the `ready` line and dictation should work.
2. Or, while it runs, check for open network connections (no output means none):
   ```bash
   lsof -a -i -p $(pgrep -f "python dictate.py")
   ```

## Configuration reference

| Setting | Where | Default |
|---|---|---|
| ASR model | `ASR_REPO` in `dictate.py` or `ASR` env var | `mlx-community/parakeet-tdt-0.6b-v2` |
| Cleanup LLM | `LLM_REPO` in `dictate.py` or `LLM` env var | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| Pull policy | `PULL` env var | `missing` |
| Tone | `tone.txt` or `TONE` env var | `casual` |
| Custom dictionary | `dictionary.txt` or `DICTIONARY` env var | none |
| Idle timeout before auto-exit | `IDLE_TIMEOUT` in `dictate.py` | 30 min |
| Cleanup instructions | `SYSTEM` in `dictate.py` | Fix punctuation/casing; remove fillers, stutters, and explicit self-corrections; never change wording |
| Cleanup examples | `FEWSHOT` in `dictate.py` | 4 raw -> formatted pairs that show wording kept verbatim |
| Dictionary terms sent to the LLM | `MAX_DICT_TERMS_IN_PROMPT` in `dictate.py` | 200 |
| Minimum recording length | `SR * 0.3` in `stop()` | 0.3 s |

To change cleanup behavior (for example, to keep fillers or format output differently), edit both `SYSTEM` and `FEWSHOT`. The examples are sent to the LLM as earlier conversation turns, and small models follow them closely: if `SYSTEM` says to keep fillers but the examples remove them, the examples usually win.

## Troubleshooting

**`Event tap failed: grant Input Monitoring + Accessibility...`**
Add your terminal app under Input Monitoring and Accessibility, then fully restart the terminal and the script.

**fn does nothing**
- Wait for the `ready in ...` line first.
- Confirm "Press 🌐 key to" is set to Do Nothing.
- macOS can disable idle event taps. The script re-enables them automatically, but restarting it is a quick fix.

**Recording is silent / empty transcriptions**
Check Microphone permission for your terminal app. A missing permission can yield silent audio instead of an error. Reset with `tccutil reset Microphone` and press fn again to get the prompt.

**Nothing is pasted**
Check Accessibility permission, and note that secure input fields block synthetic keystrokes.

**`Could not download <repo>` at startup**
The model isn't cached and the download failed. Check your internet connection and the repo name, then run again. With `PULL=never`, a missing model also stops the script by design.

**Model load fails even though it was downloaded before**
The cached download may be incomplete (for example, interrupted). Run once with `PULL=always` to re-fetch it.

**Slow startup**
Model loading and warm-up happen before the `ready` line. Nothing responds to fn until then. The first run is slower because it downloads the models.

**High memory use**
Use the smaller default LLM (Qwen3-4B), or quit the script when you aren't dictating. Exiting frees all model memory.

**Permissions stop working after an update**
If you change terminal apps or the Python binary path (for example after `brew upgrade python` or recreating the venv), macOS treats it as a new app. Grant the permissions again.

## Privacy

All processing happens locally. The only network access is downloading model weights from Hugging Face, and only when a model is missing (or with `PULL=always`). No audio or text is sent anywhere. Because text is pasted via the system clipboard, clipboard-history apps may record dictated text.

## Limitations

- macOS on Apple Silicon only.
- English by default (Parakeet v2). See "Changing the ASR model" for other languages.
- Overwrites the clipboard on every dictation.
- Models and package APIs (`parakeet-mlx`, `mlx-lm`) change over time. Install from `requirements.txt` for a reproducible setup.
