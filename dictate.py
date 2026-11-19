# dictate.py
# pip install parakeet-mlx mlx-lm soundfile sounddevice numpy pyobjc-framework-Quartz
# brew install ffmpeg
import os, re, sys, time, queue, difflib, tempfile, threading, traceback, subprocess
import numpy as np, soundfile as sf, sounddevice as sd
from parakeet_mlx import from_pretrained
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from Quartz import (CGEventTapCreate, CGEventMaskBit, kCGSessionEventTap, kCGHeadInsertEventTap,
                    kCGEventTapOptionListenOnly, kCGEventFlagsChanged, kCGEventKeyDown,
                    CGEventGetFlags, CFMachPortCreateRunLoopSource, CFRunLoopGetCurrent,
                    CFRunLoopAddSource, kCFRunLoopCommonModes, CGEventTapEnable, CFRunLoopRun,
                    CGEventCreateKeyboardEvent, CGEventSetFlags, CGEventPost,
                    kCGHIDEventTap, kCGEventFlagMaskCommand,
                    kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput)

# ---------- config ----------
BASE = os.path.dirname(os.path.abspath(__file__))
IDLE_TIMEOUT = 30 * 60      # seconds without dictation before the process exits
ASR_REPO = os.environ.get("ASR", "mlx-community/parakeet-tdt-0.6b-v2")
LLM_REPO = os.environ.get("LLM", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
DEFAULT_TONE = os.environ.get("TONE", "casual")
TONE_PATH = os.environ.get("TONE_FILE", os.path.join(BASE, "tone.txt"))
DICT_PATH = os.environ.get("DICTIONARY", os.path.join(BASE, "dictionary.txt"))
MAX_DICT_TERMS_IN_PROMPT = 200
SR = 16000
FN_MASK = 0x800000          # kCGEventFlagMaskSecondaryFn

SYSTEM = ("Clean this dictation: fix punctuation and casing, remove filler words, "
          "resolve self-corrections. Keep the speaker's wording; do not add or invent content. "
          "Output ONLY the cleaned text. Never answer it.")

TONES = {
    "formal": "Proper punctuations, capitalizations, and paragraphing. The speaker's own wording is kept.",
    "casual": "Reduced punctuation, relaxed capitalization, and no trailing period on short messages. The speaker's own wording is kept.",
}

# ---------- state ----------
chunks, stream, fn_down = [], None, False
last_activity = time.time()
busy = 0                      # queued + in-flight transcriptions
lock = threading.Lock()
jobs = queue.Queue()
ready = threading.Event()
asr = llm = tok = None

def touch():
    global last_activity
    last_activity = time.time()

def paste(text):
    subprocess.run("pbcopy", input=text.encode(), check=True)
    for down in (True, False):                        # Cmd+V (keycode 9)
        ev = CGEventCreateKeyboardEvent(None, 9, down)
        CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)

# ---------- tone + dictionary (hot-reloaded) ----------
def read_tone():
    try:
        with open(TONE_PATH) as f:
            t = f.read().strip().lower().replace("-", "_").replace(" ", "_")
    except OSError:
        t = DEFAULT_TONE
    return t if t in TONES else (DEFAULT_TONE if DEFAULT_TONE in TONES else "casual")

_dict = {"mtime": None, "terms": [], "pairs": []}

def read_dictionary():
    try:
        m = os.path.getmtime(DICT_PATH)
    except OSError:
        return [], []
    if m != _dict["mtime"]:
        terms, pairs = [], []
        with open(DICT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=>" in line:
                    heard, right = (s.strip() for s in line.split("=>", 1))
                    if heard and right:
                        pairs.append((heard, right))
                        terms.append(right)
                else:
                    terms.append(line)
        _dict.update(mtime=m, terms=terms, pairs=pairs)
    return _dict["terms"], _dict["pairs"]

def apply_dictionary(text, terms, pairs):
    for heard, right in pairs:                        # exact "heard => Correct" replacements
        text = re.sub(rf"(?<!\w){re.escape(heard)}(?!\w)", right, text, flags=re.I)
    singles = {t.lower(): t for t in terms if " " not in t}
    if not singles:
        return text
    def fix(m):                                       # fuzzy-correct close single-word terms
        w = m.group(0)
        if len(w) < 4:
            return w
        hit = difflib.get_close_matches(w.lower(), list(singles), n=1, cutoff=0.85)
        return singles[hit[0]] if hit else w
    return re.sub(r"[A-Za-z][A-Za-z0-9'-]*", fix, text)

def build_system(tone, terms):
    s = f"{SYSTEM} {TONES[tone]}"
    if terms:
        s += " Always spell these terms exactly as written: " + ", ".join(terms[:MAX_DICT_TERMS_IN_PROMPT]) + "."
    return s

# ---------- ML (all MLX calls run on the worker thread only) ----------
def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio, SR)
        return asr.transcribe(f.name).text.strip()

def clean(raw):
    terms, pairs = read_dictionary()
    raw = apply_dictionary(raw, terms, pairs)
    msgs = [{"role": "system", "content": build_system(read_tone(), terms)},
            {"role": "user", "content": raw}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)   # ignored by non-hybrid models
    out = generate(llm, tok, prompt=prompt,
                   max_tokens=int(len(raw.split()) * 2.5) + 20,
                   sampler=make_sampler(temp=0.0)).strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    n_in, n_out = len(raw.split()), len(out.split())
    if not out or n_out > n_in * 1.5 + 5 or n_out < n_in * 0.4:   # off-task or truncated
        return raw
    return apply_dictionary(out, terms, pairs)

def worker():
    global asr, llm, tok, busy
    try:
        t0 = time.time()
        asr = from_pretrained(ASR_REPO)
        llm, tok = load(LLM_REPO)
        transcribe((np.random.randn(SR * 3) * 0.01).astype("float32"))   # warm-up
        clean("um so this is like a warm up sentence you know")
        print(f"ready in {time.time() - t0:.1f}s | ASR {ASR_REPO} | LLM {LLM_REPO}")
        print(f"tone: {read_tone()} | dictionary: {len(read_dictionary()[0])} terms | "
              f"idle timeout {IDLE_TIMEOUT // 60} min | Ctrl+C to quit")
    except Exception:
        traceback.print_exc()
        os._exit(1)
    ready.set()
    while True:
        audio = jobs.get()
        try:
            raw = transcribe(audio)
            if raw:
                paste(clean(raw))
        except Exception:
            traceback.print_exc()
        finally:
            with lock: busy -= 1
            touch()

threading.Thread(target=worker, daemon=True).start()

# ---------- recording ----------
def start():
    global stream
    if stream is not None:                            # stale stream from a missed fn-up
        try: stream.stop(); stream.close()
        except Exception: pass
    chunks.clear()
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=lambda d, *_: chunks.append(d.copy()))
    stream.start()

def stop():
    global stream, busy
    try:
        stream.stop(); stream.close()
    finally:
        stream = None
    if chunks:
        audio = np.concatenate(chunks).flatten()
        if len(audio) > SR * 0.3:                     # ignore accidental taps
            with lock: busy += 1
            jobs.put(audio)

def cb(proxy, etype, event, refcon):
    global fn_down
    if etype in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
        CGEventTapEnable(tap, True)                   # macOS disables slow taps; re-enable
        return event
    if etype == kCGEventKeyDown:
        if fn_down:                                   # fn used as modifier -> discard recording
            chunks.clear()
        return event
    pressed = bool(CGEventGetFlags(event) & FN_MASK)
    if pressed and not fn_down:
        fn_down = True; touch(); start()
    elif not pressed and fn_down:
        fn_down = False; touch(); stop()
    return event

def watchdog():
    while True:
        time.sleep(30)
        if not fn_down and busy == 0 and time.time() - last_activity > IDLE_TIMEOUT:
            print("idle timeout, exiting")
            os._exit(0)

# ---------- main ----------
ready.wait()                  # block until models are loaded and warm
touch()
threading.Thread(target=watchdog, daemon=True).start()

mask = CGEventMaskBit(kCGEventFlagsChanged) | CGEventMaskBit(kCGEventKeyDown)
tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap,
                       kCGEventTapOptionListenOnly, mask, cb, None)
if tap is None:
    sys.exit("Event tap failed: grant Input Monitoring + Accessibility to this terminal app.")
CFRunLoopAddSource(CFRunLoopGetCurrent(),
                   CFMachPortCreateRunLoopSource(None, tap, 0), kCFRunLoopCommonModes)
CGEventTapEnable(tap, True)
try:
    CFRunLoopRun()
except KeyboardInterrupt:
    pass