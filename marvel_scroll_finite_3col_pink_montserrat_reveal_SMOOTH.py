#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marvel_scroll_finite_3col_pink_montserrat_reveal_SMOOTHv2_CUDA.py
-----------------------------------------------------------------
Aprovecha **GPU** para la generación de frames (si hay CUDA) y NVENC para el encode.

Qué acelera en GPU (PyTorch CUDA):
- Composición de paneles por frame (reveal superior/inferior)
- Ensamblado de la cinta completa
- Recorte subpíxel del viewport con `grid_sample` (scroll suave)

Si no hay CUDA, cae a CPU automáticamente (misma salida).

Requisitos:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # (ajusta a tu CUDA)
  pip install moviepy pillow numpy imageio-ffmpeg
"""
import os, glob, argparse, subprocess, math
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip
import re

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

# ---------- utils ----------
def parse_size(s: str) -> Tuple[int,int]:
    w,h = s.lower().split('x'); return int(w), int(h)

def load_font(font_path: Optional[str], size: int):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [font_path] if font_path else []
    cands.append(os.path.join(here, "Montserrat-Bold.ttf"))
    for p in cands:
        try:
            if p and os.path.exists(p):
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def fit_cover(img: Image.Image, target: Tuple[int,int]) -> Image.Image:
    W,H = target
    img = img.convert("RGB")
    w,h = img.size
    s = max(W/max(1,w), H/max(1,h))
    nw,nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
    rsz = img.resize((nw,nh), Image.LANCZOS)
    x = (nw - W)//2; y = (nh - H)//2
    return rsz.crop((x,y,x+W,y+H))

def wrap_text(draw, text, font, max_w):
    words = (text or "").split()
    lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textbbox((0,0), t, font=font)[2] <= max_w or not cur:
            cur=t
        else:
            lines.append(cur); cur=w
    if cur: lines.append(cur)
    return lines

def natural_key(s):
    # Extrae números para ordenar naturalmente
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

# ---------- layout ----------
def calc_heights(H:int):
    name_h  = int(H*0.125)
    label_h = int(H*0.095)
    rem = H - name_h - label_h
    top_h = rem // 2
    bot_h = rem - top_h
    return top_h, name_h, label_h, bot_h

def draw_gradient_bar(draw: ImageDraw.ImageDraw, x1,y1,x2,y2, color_left=(255,0,127), color_right=(255,77,184)):
    W = x2-x1; H = y2-y1
    for i in range(W):
        t = 0 if W<=1 else i/(W-1)
        r = int(round(color_left[0]*(1-t) + color_right[0]*t))
        g = int(round(color_left[1]*(1-t) + color_right[1]*t))
        b = int(round(color_left[2]*(1-t) + color_right[2]*t))
        draw.line([(x1+i, y1), (x1+i, y2-1)], fill=(r,g,b))

def render_bars_text(panel_w:int, H:int, name:str, font_path:Optional[str], sep_right:int) -> np.ndarray:
    top_h, name_h, label_h, bot_h = calc_heights(H)
    img = Image.new("RGB",(panel_w,H),(0,0,0))
    d = ImageDraw.Draw(img)
    draw_gradient_bar(d, 0, top_h, panel_w, top_h+name_h, (255,0,127), (255,77,184))
    d.rectangle([0, top_h+name_h, panel_w-1, top_h+name_h+label_h-1], fill=(18,18,18))
    pad = int(panel_w*0.06)
    # Ajusta el tamaño de fuente para el nombre
    font_main, lines = fit_font_size(d, (name or "").upper(), font_path, panel_w - 2*pad, name_h)
    sizes=[]; tot_h=0
    for ln in lines:
        bbox = d.textbbox((0,0), ln, font=font_main, stroke_width=2)
        w = bbox[2]-bbox[0]; h=bbox[3]-bbox[1]
        sizes.append((w,h)); tot_h += h
    gap = max(1, int(font_main.size*0.06)); tot_h += gap*(len(lines)-1)
    cy = top_h + (name_h - tot_h)//2
    for ln,(lw,lh) in zip(lines, sizes):
        cx = (panel_w - lw)//2
        d.text((cx,cy), ln, font=font_main, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
        cy += lh + gap
    # Label fijo
    font_lbl  = load_font(font_path, max(9, int(label_h*0.60)))
    label = "SUPERMAN (2025)"
    bbox = d.textbbox((0,0), label, font=font_lbl, stroke_width=2)
    lw=bbox[2]-bbox[0]; lh=bbox[3]-bbox[1]
    lb_y1 = top_h+name_h
    d.text(((panel_w-lw)//2, lb_y1 + (label_h-lh)//2), label, font=font_lbl,
           fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
    if sep_right>0:
        d.rectangle([panel_w-sep_right, 0, panel_w-1, H-1], fill=(0,0,0))
    return np.array(img, dtype=np.uint8)

# ---------- GPU helpers ----------
def to_tensor_uint8(img_np: np.ndarray, device) -> "torch.Tensor":
    t = torch.from_numpy(img_np).to(device=device, dtype=torch.uint8)
    t = t.permute(2,0,1).unsqueeze(0).to(dtype=torch.float32) / 255.0
    return t

def from_tensor_uint8(t: "torch.Tensor") -> np.ndarray:
    t = (t.clamp(0,1)*255.0).to(dtype=torch.uint8)
    t = t.squeeze(0).permute(1,2,0).contiguous()
    return t.cpu().numpy()

def blank_tensor(H:int,W:int, device):
    return torch.zeros((1,3,H,W), device=device, dtype=torch.float32)

def compose_panel_cuda(panel_w:int, H:int, bars_t, top_t, bot_t, progress: float, sep_right:int, device):
    top_h, name_h, label_h, bot_h = calc_heights(H)
    prog = max(0.0, min(1.0, progress))
    canvas = blank_tensor(H, panel_w, device)
    k_top = int(round(top_h * prog))
    if k_top>0:
        src = top_t[..., -k_top:, :]  # <-- Asegura shape [1,3,k_top,panel_w]
        y1 = top_h - k_top
        canvas[..., y1:top_h, :] = src
    k_bot = int(round(bot_h * prog))
    if k_bot>0:
        y0 = H - bot_h
        src = bot_t[..., :k_bot, :]   # <-- Asegura shape [1,3,k_bot,panel_w]
        canvas[..., y0:y0+k_bot, :] = src
    canvas = bars_t + canvas * (bars_t.eq(0).float())
    return canvas

def build_grid_for_crop(W:int,H:int, x_offset:float, strip_W:int, device):
    xs = torch.linspace(0, W-1, W, device=device)
    ys = torch.linspace(0, H-1, H, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Xs = X + x_offset
    xn = 2*(Xs / (strip_W-1)) - 1
    yn = 2*(Y  / (H-1)) - 1
    grid = torch.stack([xn, yn], dim=-1).unsqueeze(0)
    return grid

# ---------- GPU detection ----------
def gpu_device():
    if (torch is not None) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if torch is not None else None

# ---------- easing ----------
def ease(t: float, mode: str) -> float:
    t = max(0.0, min(1.0, t))
    if mode == "sine":
        import math
        return 0.5 - 0.5*math.cos(math.pi*t)
    return t

# ---------- ffmpeg ----------
def ffmpeg_has_encoder(name: str) -> bool:
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE","ffmpeg")
    try:
        out = subprocess.check_output([exe, "-hide_banner", "-encoders"],
                                      stderr=subprocess.STDOUT, universal_newlines=True)
        return name in out
    except Exception:
        return False

def pick_best_gpu_codec(preferred: Optional[str]=None):
    if preferred:
        if preferred.endswith("_nvenc"):
            return preferred, ["-cq","23","-pix_fmt","yuv420p","-movflags","+faststart"], "p4"
        return preferred, ["-crf","18","-pix_fmt","yuv420p","-movflags","+faststart"], "medium"
    for cod,params,preset in [
        ("h264_nvenc", ["-cq","23","-pix_fmt","yuv420p","-movflags","+faststart"], "p4"),
        ("hevc_nvenc", ["-cq","23","-pix_fmt","yuv420p","-movflags","+faststart"], "p4"),
        ("av1_nvenc",  ["-cq","28","-pix_fmt","yuv420p","-movflags","+faststart"], "p5"),
    ]:
        if ffmpeg_has_encoder(cod): return cod, params, preset
    return "libx264", ["-crf","18","-pix_fmt","yuv420p","-movflags","+faststart"], "medium"

def pick_best_gpu_codec_force():
    # Fuerza el uso de NVENC y verifica disponibilidad
    if ffmpeg_has_encoder("h264_nvenc"):
        return "h264_nvenc", ["-cq","23","-pix_fmt","yuv420p","-movflags","+faststart"], "p4"
    print("[ERROR] No se encontró soporte NVENC en FFmpeg. Instala FFmpeg con soporte para tu GPU Nvidia.")
    raise SystemExit("No se puede continuar sin NVENC.")

# ---------- main ----------
def main():
    print("Iniciando script...")  # <-- Agrega esto
    ap = argparse.ArgumentParser(description="Scroller finito 3 cols - GPU frames + NVENC")
    ap.add_argument("--size", type=parse_size, default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--sep_px", type=int, default=2)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--scroll_speed", type=float, default=20.0)
    ap.add_argument("--tail_pause", type=float, default=3.0)
    ap.add_argument("--codec", type=str, default=None)
    ap.add_argument("--font_path", type=str, default=None)
    ap.add_argument("--reveal_dur", type=float, default=0.6)
    ap.add_argument("--ease", type=str, default="linear", choices=["linear","sine"])
    args = ap.parse_args()

    W,H = args.size
    panel_w = W // max(1, int(args.cols))

    root = os.path.dirname(os.path.abspath(__file__))
    arriba = sorted(glob.glob(os.path.join(root,"arriba","*")), key=natural_key)
    abajo  = sorted(glob.glob(os.path.join(root,"abajo","*")), key=natural_key)
    names_file = os.path.join(root,"nombres.txt")
    if not os.path.exists(names_file):
        raise SystemExit("Falta nombres.txt")
    with open(names_file,encoding="utf-8") as f:
        nombres = [ln.strip() for ln in f if ln.strip()]
    if not (len(arriba)==len(abajo)==len(nombres)):
        raise SystemExit("arriba, abajo y nombres.txt deben tener la MISMA cantidad")
    N = len(nombres)

    top_h, name_h, label_h, bot_h = calc_heights(H)

    bars_list = []
    top_np = []
    bot_np = []
    for i in range(N):
        sep_right = args.sep_px if i!=N-1 else 0
        bars_list.append(render_bars_text(panel_w, H, nombres[i], args.font_path, sep_right))
        with Image.open(arriba[i]) as T:
            top_np.append(np.array(fit_cover(T,(panel_w, top_h)).convert("RGB")))
        with Image.open(abajo[i]) as B:
            bot_np.append(np.array(fit_cover(B,(panel_w, bot_h)).convert("RGB")))

    device = gpu_device()
    if device is not None and device.type == "cuda":
        print("[GPU] CUDA activada:", torch.cuda.get_device_name(0))
    elif device is None:
        print("[CPU] PyTorch no está instalado. Solo CPU disponible.")
    else:
        print("[CPU] Usando CPU para generación de frames.")

    bars_t = [to_tensor_uint8(b, device) for b in bars_list]
    top_t  = [to_tensor_uint8(x, device) for x in top_np]
    bot_t  = [to_tensor_uint8(x, device) for x in bot_np]

    offsets = []
    xoff = 0
    for i in range(N):
        offsets.append(xoff)
        xoff += panel_w
        if i!=N-1:
            xoff += args.sep_px + args.gap
    strip_w = xoff
    total_scroll = max(0, strip_w - W)

    def trigger_time(i: int) -> float:
        x_trigger = max(0, offsets[i] - (W - panel_w))
        return x_trigger / max(1e-6, args.scroll_speed)
    triggers = [trigger_time(i) for i in range(N)]
    end_time = total_scroll / max(1e-6, args.scroll_speed)
    duration = end_time + args.tail_pause

    def make_frame(t):
        prog_list = []
        for i in range(N):
            rel = t - triggers[i]
            prog = 0.0 if rel<=0 else min(1.0, rel / max(1e-6, args.reveal_dur))
            prog_list.append(prog)
        strip = torch.zeros((1,3,H,strip_w), device=device, dtype=torch.float32)
        for i in range(N):
            panel = compose_panel_cuda(panel_w, H, bars_t[i], top_t[i], bot_t[i],
                                       prog_list[i], sep_right=(args.sep_px if i!=N-1 else 0), device=device)
            x0 = offsets[i]
            strip[..., :, x0:x0+panel_w] = panel
        t_lin = min(end_time, max(0.0, t))
        u = ease(t_lin / max(1e-6, end_time), args.ease)
        x = u * total_scroll
        grid = build_grid_for_crop(W, H, x, strip_w, device)
        viewport = F.grid_sample(strip, grid, mode="bilinear", align_corners=True)
        img_np = from_tensor_uint8(viewport)
        return img_np

    clip = VideoClip(make_frame, duration=duration)
    codec, params, preset = pick_best_gpu_codec_force()  # <-- Fuerza NVENC
    out_path = os.path.join(root, "marvel_scroll_finite_3col_pink_montserrat_reveal_SMOOTHv2_CUDA.mp4")
    print(f"[export] size={W}x{H} fps={args.fps} speed={args.scroll_speed}px/s duration={duration:.2f}s ease={args.ease} device={device}")
    print(f"[INFO] Usando codec: {codec}")
    clip.write_videofile(out_path, fps=args.fps, codec=codec, audio=False, preset=preset, ffmpeg_params=params)
    print(f"[OK] Generado: {out_path}")

def fit_font_size(draw, text, font_path, max_w, max_h, min_size=10, max_size=100):
    # Busca el tamaño máximo de fuente que permita que el texto quepa en el área
    for size in range(max_size, min_size-1, -1):
        font = load_font(font_path, size)
        lines = wrap_text(draw, text, font, max_w)
        heights = [draw.textbbox((0,0), ln, font=font, stroke_width=2)[3] for ln in lines]
        total_h = sum(heights) + max(1, int(size*0.06))*(len(lines)-1)
        widths = [draw.textbbox((0,0), ln, font=font, stroke_width=2)[2] for ln in lines]
        if max(widths) <= max_w and total_h <= max_h:
            return font, lines
    # Si no cabe, regresa el mínimo
    font = load_font(font_path, min_size)
    lines = wrap_text(draw, text, font, max_w)
    return font, lines

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()

