import json, time, urllib.request, sys
U = "http://127.0.0.1:18188"
BASE = ("a woman, light makeup, soft smudged brown eyeliner, natural lip tint, light freckles, dark brown hair with bangs, "
        "slim athletic, full perky round breasts, deep cleavage, thin black choker, dark nails, ")
END = ("in her lived-in goth bedroom at night, lived-in messy room, moody low-light snapshot, soft dim ambient room light, "
       "muted desaturated colors, cool soft shadows, warm lamp glow and fairy lights in the background, casual private snapshot "
       "she quickly took to send to her boyfriend, unposed, imperfect framing, unretouched phone photo, slight grain")
NEG = ("nipples, areola, topless, nude, bare breasts, jeans, heavy makeup, bright daylight, overexposed, phone case, phone held to ear, "
       "child, teen, young-looking, old, cartoon, anime, 3d render, plastic skin, airbrushed, deformed hands, extra fingers, "
       "extra arms, extra legs, watermark, text")
T2I = BASE + "front camera selfie, phone out of frame, chest up close up, leaning in toward the camera, wearing a faded charcoal low-cut ribbed cami, " + END
EDIT_A = ("Keep the woman from image 1 exactly: same face, same eyes, same freckles, same dark brown hair with bangs, same thin black choker. "
          "Show her in a new photo: a front camera selfie from slightly above, chest up, leaning in toward the camera with a relaxed soft half smile, "
          "wearing a faded charcoal oversized knit sweater slipping off one shoulder over a low-cut black cami, fully clothed, " + END)
EDIT_B = ("Keep the woman from image 1 exactly: same face, same eyes, same freckles, same dark brown hair with bangs, same thin black choker. "
          "Show her in a new photo: a mirror selfie, standing in front of a full-length mirror, holding a space black iPhone 14 Pro with no case at chest height, "
          "her face visible, wearing a washed black fitted long-sleeve top with a deep scoop neck and faded charcoal sweatpants, fully clothed, " + END)

def graph(prompt, seed, prefix, w=None, h=None, ref=None):
    g = {
     "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "chatrlabs_qwen21_fp8_comfy.safetensors", "weight_dtype": "default"}},
     "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
     "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
     "4": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
     "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": NEG, "resolution": 1024, "vae": ["3", 0]}},
     "7": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2] if ref else ["6", 0],
            "seed": seed, "steps": 35, "cfg": 5.0, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
     "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
     "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }
    if ref:
        g["10"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        g["5"]["inputs"]["images.image_1"] = ["10", 0]
    else:
        g["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    return g

def run(g, name):
    req = urllib.request.Request(U + "/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(name, "SUBMIT_ERROR", e.read().decode()[:3000], flush=True); return
    t0 = time.time()
    while True:
        time.sleep(3)
        h = json.loads(urllib.request.urlopen(U + "/history/" + pid).read())
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            st = h[pid]["status"]
            msgs = {m[0]: m[1].get("timestamp") for m in st.get("messages", [])}
            ex = None
            if "execution_start" in msgs and ("execution_success" in msgs):
                ex = (msgs["execution_success"] - msgs["execution_start"]) / 1000
            print(name, st.get("status_str"), "wall=%.1fs" % (time.time() - t0), "exec=%ss" % ex, json.dumps(st.get("messages"))[-1500:] if st.get("status_str") != "success" else "", flush=True)
            return

jobs = [
  ("t2i_a", graph(T2I, 2101, "q21_t2i_a", 832, 1216)),
  ("edit_a", graph(EDIT_A, 2102, "q21_edit_a", ref="qwen21_ref.jpg")),
  ("edit_b", graph(EDIT_B, 2103, "q21_edit_b", ref="qwen21_ref.jpg")),
  ("t2i_b", graph(T2I, 2104, "q21_t2i_b", 832, 1216)),
]
only = sys.argv[1:] 
for n, g in jobs:
    if only and n not in only: continue
    run(g, n)
print("ALL_DONE", flush=True)
