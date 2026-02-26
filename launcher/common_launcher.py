import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os
from pathlib import Path
import platform
import random

# ── THEME COLORS ──
BG_TOP = "#7C6EE6"
BG_BOTTOM = "#5B2C83"
BG_MID = "#6B4EA8"
BTN_PINK = "#E879F9"
BTN_RED = "#F43F5E"
BTN_BLUE = "#60A5FA"
BTN_HOVER = "#A78BFA"
TEXT_WHITE = "#FFFFFF"
TEXT_SOFT = "#E5D9FF"
CARD_BG = "#5F4BC4"

# ── Responsive breakpoint (px) ──
MOBILE_BREAKPOINT = 620


# ── UI HELPERS ──
def draw_rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def create_rounded_button(parent, text, bg_color, command,
                          font_size=12, width=260, height=50):
    """Creates a rounded-rectangle button on a Canvas."""
    canvas = tk.Canvas(parent, width=width, height=height,
                       bg=parent["bg"], highlightthickness=0)
    r = min(24, height // 2)
    rect = draw_rounded_rect(canvas, 2, 2, width - 2, height - 2, r,
                             fill=bg_color, outline=bg_color)
    canvas.create_text(width // 2, height // 2, text=text, fill="white",
                       font=("Segoe UI", font_size, "bold"))

    def on_enter(e):
        canvas.itemconfig(rect, fill=BTN_HOVER, outline=BTN_HOVER)

    def on_leave(e):
        canvas.itemconfig(rect, fill=bg_color, outline=bg_color)

    def on_click(e):
        command()

    canvas.bind("<Enter>", on_enter)
    canvas.bind("<Leave>", on_leave)
    canvas.bind("<Button-1>", on_click)
    return canvas


# ── GRADIENT CANVAS ──
class GradientFrame(tk.Canvas):
    def __init__(self, parent, color1, color2, **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.color1 = color1
        self.color2 = color2
        self._cache_h = 0
        self.bind("<Configure>", self._draw_gradient)

    def _draw_gradient(self, event=None):
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2 or h == self._cache_h:
            return
        self._cache_h = h
        self.delete("gradient")
        r1, g1, b1 = self.winfo_rgb(self.color1)
        r2, g2, b2 = self.winfo_rgb(self.color2)
        for i in range(h):
            nr = int(r1 + (r2 - r1) * i / h)
            ng = int(g1 + (g2 - g1) * i / h)
            nb = int(b1 + (b2 - b1) * i / h)
            color = f"#{nr >> 8:02x}{ng >> 8:02x}{nb >> 8:02x}"
            self.create_line(0, i, w, i, fill=color, tags=("gradient",))


# ── WELCOME MESSAGES & TIPS ──
WELCOME_GREETINGS = [
    "Welcome back! Ready to explore?",
    "Hey there! Let Moody brighten your day.",
    "Hello, friend! Your AI companion awaits.",
    "Good to see you! Let's get started.",
    "Welcome! Moody is here to help.",
]

TIPS = [
    "💡 Tip: Use the Emotion module to control your PC with facial expressions.",
    "💡 Tip: Hand gestures let you navigate without touching the keyboard.",
    "💡 Tip: Moody adapts to your mood and suggests content for you.",
    "💡 Tip: Try resizing this window — the layout adapts automatically!",
    "💡 Tip: You can launch modules and the launcher will close automatically.",
]

FEATURES = [
    ("🎭", "Emotion Detection", "Real-time facial expression analysis with AI-powered recognition."),
    ("✋", "Hand Gestures", "Navigate and control your PC with intuitive hand movements."),
    ("🧠", "Smart Analytics", "Track your emotional patterns and get personalized insights."),
]


# ══════════════════════════════════════════════
#  MAIN LAUNCHER APP  (responsive + hamburger)
# ══════════════════════════════════════════════
class MoodyLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MOODY")

        # DPI awareness (Windows)
        try:
            if platform.system() == "Windows":
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        # ── Responsive: resizable, min-size for mobile ──
        self.geometry("900x750")
        self.minsize(360, 520)
        self.resizable(True, True)

        # Paths
        self.base_dir = Path(__file__).resolve().parent.parent
        venv_python = self.base_dir / ".venv" / "Scripts" / "python.exe"
        self.python = str(venv_python if venv_python.exists() else Path(sys.executable))
        self.procs = {}

        # Image holders
        self.hero_img_tk = None
        self.hero_img_small = None

        # Layout state
        self._is_mobile = False
        self._tip_index = 0

        # Build UI
        self._build_ui()

        # React to resize
        self.bind("<Configure>", self._on_resize)

    # ─────────────────────────
    #  BUILD ENTIRE UI
    # ─────────────────────────
    def _build_ui(self):
        # Root gradient background
        self.bg = GradientFrame(self, BG_TOP, BG_BOTTOM)
        self.bg.pack(fill="both", expand=True)

        # ── Top bar ──
        self.topbar = tk.Frame(self.bg, bg=BG_TOP, height=50)
        self.topbar.pack(fill="x", side="top")
        self.topbar.pack_propagate(False)

        # Topbar title
        self.topbar_title = tk.Label(
            self.topbar, text="MOODY", bg=BG_TOP, fg=TEXT_WHITE,
            font=("Segoe UI", 14, "bold"),
        )
        self.topbar_title.pack(side="left", padx=14, pady=6)

        # ── Scrollable content area ──
        self.scroll_canvas = tk.Canvas(self.bg, bg=BG_TOP, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.bg, orient="vertical",
                                       command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(self.scroll_canvas, bg=BG_TOP)
        self.content_window = self.scroll_canvas.create_window(
            (0, 0), window=self.content, anchor="n",
        )

        self.content.bind("<Configure>",
                          lambda e: self.scroll_canvas.configure(
                              scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", self._on_canvas_resize)

        # Mouse-wheel scrolling
        self.scroll_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

        # ── Populate content ──
        self._build_header()
        self._build_hero_image()
        self._build_welcome_section()
        self._build_get_started_button()
        self._build_features_section()
        self._build_tip_bar()
        self._build_footer()

        # Start rotating tips
        self._rotate_tip()

    # ── Canvas width tracking for centering ──
    def _on_canvas_resize(self, event):
        self.scroll_canvas.itemconfig(self.content_window, width=event.width)

    # ─────────────────────────
    #  GET STARTED BUTTON
    # ─────────────────────────
    def _build_get_started_button(self):
        """Prominent Get Started button to launch the emotion module."""
        btn_frame = tk.Frame(self.content, bg=BG_TOP)
        btn_frame.pack(pady=(15, 10))

        create_rounded_button(
            btn_frame, "🚀  Get Started", BTN_PINK,
            self.launch_emotion, font_size=15, width=300, height=56,
        ).pack()

    # ─────────────────────────
    #  RESPONSIVE RESIZE LOGIC
    # ─────────────────────────
    def _on_resize(self, event=None):
        w = self.winfo_width()
        mobile = w < MOBILE_BREAKPOINT

        if mobile == self._is_mobile:
            return  # no layout change needed
        self._is_mobile = mobile

        if mobile:
            # Shrink hero image for narrow window
            self._show_hero(small=True)
        else:
            # Full-size hero for wide window
            self._show_hero(small=False)

    # ─────────────────────────
    #  CONTENT BUILDERS
    # ─────────────────────────
    def _build_header(self):
        """Title + subtitle."""
        hdr = tk.Frame(self.content, bg=BG_TOP)
        hdr.pack(pady=(30, 0))

        tk.Label(
            hdr, text="MOODY", fg=TEXT_WHITE, bg=BG_TOP,
            font=("Segoe UI", 42, "bold"),
        ).pack()

        tk.Label(
            hdr, text="AI Control System", fg=TEXT_SOFT, bg=BG_TOP,
            font=("Segoe UI", 13),
        ).pack(pady=(2, 0))

    def _build_hero_image(self):
        """Load hero image in two sizes for responsive swap."""
        self.hero_label = tk.Label(self.content, bg=BG_TOP)
        self.hero_label.pack(pady=(15, 10))

        try:
            from PIL import Image, ImageTk
            img_path = self.base_dir / "assets" / "cat.png"
            if img_path.exists():
                pil_img = Image.open(str(img_path))
                self.hero_img_tk = ImageTk.PhotoImage(pil_img.resize((260, 300)))
                self.hero_img_small = ImageTk.PhotoImage(pil_img.resize((140, 160)))
                self.hero_label.config(image=self.hero_img_tk)
        except Exception:
            self.hero_label.config(
                text="🐱", font=("Segoe UI Emoji", 64), fg=TEXT_WHITE,
            )

    def _show_hero(self, small=False):
        if small and self.hero_img_small:
            self.hero_label.config(image=self.hero_img_small)
        elif self.hero_img_tk:
            self.hero_label.config(image=self.hero_img_tk)

    def _build_welcome_section(self):
        """Rich welcome messages."""
        sec = tk.Frame(self.content, bg=BG_TOP)
        sec.pack(pady=(5, 10), padx=30, fill="x")

        # Random greeting
        greeting = random.choice(WELCOME_GREETINGS)
        tk.Label(
            sec, text=greeting, fg=TEXT_WHITE, bg=BG_TOP,
            font=("Segoe UI", 16, "bold"), wraplength=700, justify="center",
        ).pack(pady=(0, 8))

        # Long description
        desc_lines = (
            "Experience seamless computer control through emotion recognition,\n"
            "hand gestures, and natural voice commands — all powered by AI.\n\n"
            "Moody understands how you feel and lets you interact\n"
            "with your PC in a whole new way.\n"
            "Click Get Started to begin your journey!"
        )
        tk.Label(
            sec, text=desc_lines, fg=TEXT_SOFT, bg=BG_TOP,
            font=("Segoe UI", 10), justify="center", wraplength=650,
        ).pack()

    def _build_features_section(self):
        """Feature highlight cards."""
        wrapper = tk.Frame(self.content, bg=BG_TOP)
        wrapper.pack(pady=(10, 10), padx=20, fill="x")

        tk.Label(
            wrapper, text="What Moody Can Do", fg=TEXT_WHITE, bg=BG_TOP,
            font=("Segoe UI", 15, "bold"),
        ).pack(pady=(0, 10))

        # Cards container
        self.features_container = tk.Frame(wrapper, bg=BG_TOP)
        self.features_container.pack(fill="x")

        for icon, title, desc in FEATURES:
            card = tk.Frame(self.features_container, bg=CARD_BG, bd=0,
                            highlightbackground="#7C6EE6", highlightthickness=1)
            card.pack(fill="x", pady=5, padx=10, ipady=8)

            inner = tk.Frame(card, bg=CARD_BG)
            inner.pack(fill="x", padx=14, pady=8)

            tk.Label(
                inner, text=f"{icon}  {title}", fg=TEXT_WHITE, bg=CARD_BG,
                font=("Segoe UI", 12, "bold"), anchor="w",
            ).pack(fill="x")

            tk.Label(
                inner, text=desc, fg=TEXT_SOFT, bg=CARD_BG,
                font=("Segoe UI", 9), anchor="w", wraplength=600, justify="left",
            ).pack(fill="x", pady=(3, 0))

    def _build_tip_bar(self):
        """Rotating tips bar."""
        tip_frame = tk.Frame(self.content, bg=CARD_BG)
        tip_frame.pack(fill="x", padx=30, pady=(15, 5), ipady=6)

        self.tip_label = tk.Label(
            tip_frame, text=TIPS[0], fg=TEXT_SOFT, bg=CARD_BG,
            font=("Segoe UI", 9), wraplength=600, justify="center",
        )
        self.tip_label.pack(padx=12, pady=6)

    def _rotate_tip(self):
        self._tip_index = (self._tip_index + 1) % len(TIPS)
        self.tip_label.config(text=TIPS[self._tip_index])
        self.after(6000, self._rotate_tip)

    def _build_footer(self):
        """Version / status footer."""
        ft = tk.Frame(self.content, bg=BG_TOP)
        ft.pack(fill="x", pady=(15, 25))

        tk.Label(
            ft, text="Moody v1.0  •  System Ready  •  Built with ❤️",
            fg=TEXT_SOFT, bg=BG_TOP, font=("Segoe UI", 9),
        ).pack()

    # ─────────────────────────
    #  MODULE LAUNCH ACTIONS
    # ─────────────────────────
    def launch_emotion(self):
        """Launch emotion & gesture module."""
        try:
            script = self.base_dir / "emotion_gesture" / "fullemotionmodule.py"
            if not script.exists():
                messagebox.showerror("Module Not Found",
                                     f"Cannot find emotion module:\n{script}")
                return
            if self._running("emotion"):
                messagebox.showinfo("Already Running",
                                    "Emotion & Gesture module is already running.")
                return

            proc = subprocess.Popen(
                [self.python, str(script)],
                cwd=str(script.parent),
                env=self._merged_env(),
            )
            self.procs["emotion"] = proc
            self.after(500, self.destroy)
        except Exception as e:
            messagebox.showerror("Launch Error", f"Error: {str(e)}")

    # ── HELPERS ──
    def _merged_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.base_dir) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _running(self, key):
        return key in self.procs and self.procs[key].poll() is None

    def on_close(self):
        for p in self.procs.values():
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        self.destroy()


def main():
    app = MoodyLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()