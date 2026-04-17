"""Wizard-style GUI for configuring VLM block manipulation tasks."""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import yaml
import os
import random
import math

from .constants import PALETTE, REALSENSE_AVAILABLE, ALL_BLOCK_COLORS, AREA_COLORS
from .stack_builder import StackBuilder

if REALSENSE_AVAILABLE:
    import pyrealsense2 as rs


class VLMInputGUI:
    """Wizard-style GUI for configuring VLM block manipulation tasks."""

    BLOCK_COLORS = ALL_BLOCK_COLORS
    AREA_COLORS = AREA_COLORS
    PALETTE = PALETTE

    def __init__(self, root):
        self.root = root
        self.root.title("✨ VLM Block Manipulation ✨")
        self.root.geometry("1750x820")
        self.root.resizable(False, False)
        self.root.configure(bg=self.PALETTE["bg_dark"])

        # Animation state
        self.animation_running = True
        self.particles = []
        self.glow_phase = 0

        # Wizard state
        self.current_step = 1
        self.total_steps = 3

        # Variables
        self.task_var = tk.StringVar(value="stack")
        self.stack_target_var = tk.StringVar(value="white")

        # Target click position (for clicking on image)
        self.target_click_pos = None  # (y, x) normalized 0-1000
        self.target_marker = None  # Canvas item for marker
        self.click_mode_active = False

        # White paper detection (for stack step 3)
        # List of (x, y, w, h) bounding rects in pixel coords
        self.detected_papers = []
        self.selected_paper_idx = None  # Index of selected paper

        # Sort mode: block-to-position mapping
        self.sort_block_positions = {}  # {"red": (x, y), "green": (x, y), ...}
        self.current_mapping_block = None  # Which block is being mapped
        self.sort_block_buttons = {}  # References to block selection buttons

        # Interactive widgets
        self.stack_builder = None

        # Result storage
        self.result = None

        # Camera
        self.pipeline = None
        self.camera_running = False
        self.camera_label = None
        self.camera_exposure = 450
        self.camera_contrast = 70
        self.brightness_target = 20  # target mean of non-white pixels

        # Step frames
        self.step_frames = {}
        self.controls_frame = None

        self._init_camera()
        self._create_widgets()
        self._start_camera_feed()
        self._show_step(1)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard shortcuts for camera exposure / contrast
        self.root.bind("<e>", lambda e: self._adjust_camera("exposure", 50))
        self.root.bind("<d>", lambda e: self._adjust_camera("exposure", -50))
        self.root.bind("<r>", lambda e: self._adjust_camera("contrast", 5))
        self.root.bind("<f>", lambda e: self._adjust_camera("contrast", -5))
        self.root.bind("<t>", lambda e: self._adjust_camera(
            "brightness_target", 5))
        self.root.bind("<g>", lambda e: self._adjust_camera(
            "brightness_target", -5))

    def _init_camera(self):
        """Initialize RealSense camera."""
        if not REALSENSE_AVAILABLE:
            return

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280,
                                 720, rs.format.rgb8, 30)
            self.pipeline.start(config)
            self.camera_running = True
            # set exposure
            sensor = self.pipeline.get_active_profile(
            ).get_device().query_sensors()[1]
            sensor.set_option(rs.option.exposure, self.camera_exposure)
            # set contrast
            sensor.set_option(rs.option.contrast, self.camera_contrast)

            # Warm up camera
            for _ in range(10):
                self.pipeline.wait_for_frames()
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.pipeline = None
            self.camera_running = False

    def _auto_exposure(self):
        """Automatically adjust camera exposure by metering on non-white pixels.

        Masks out bright/white pixels (>200) so they don't influence the
        exposure calculation. Targets the mean brightness of the remaining
        pixels (blocks, table, etc.) to the brightness_target value.
        """
        if not self.pipeline or not self.camera_running:
            return

        self.auto_exp_btn.config(state=tk.DISABLED, text="⏳ AUTO EXP...")
        # Pause the camera feed so we're the only consumer of frames
        self._auto_exp_paused_camera = self.camera_running
        self.camera_running = False
        self._auto_exp_state = {
            "target": self.brightness_target,
            "tolerance": 8,
            "white_threshold": 200,
            "max_iter": 25,
            "iteration": 0,
            "flush_count": 0,
            "flush_target": 15,  # frames to flush after each exposure change
            "prev_error": None,
            "sensor": self.pipeline.get_active_profile().get_device().query_sensors()[1],
        }
        self._auto_exposure_step()

    def _auto_exposure_done(self):
        """Re-enable camera feed and button after auto-exposure finishes."""
        self.auto_exp_btn.config(state=tk.NORMAL, text="☀ AUTO EXP")
        if self._auto_exp_paused_camera and self.pipeline:
            self.camera_running = True
            self._update_camera()

    def _auto_exposure_step(self):
        """One step of auto-exposure, scheduled via root.after() so the GUI stays live."""
        s = self._auto_exp_state
        if not self.pipeline:
            self._auto_exposure_done()
            return

        # Flush phase: discard stale frames so the sensor settles on the new exposure
        if s["flush_count"] < s["flush_target"]:
            try:
                self.pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                pass
            s["flush_count"] += 1
            self.root.after(40, self._auto_exposure_step)
            return

        # Measure phase – average over a few frames to reduce noise
        try:
            measurements = []
            for _ in range(3):
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
                non_white = gray[gray < s["white_threshold"]]
                if len(non_white) < 100:
                    measurements.append(float(np.mean(gray)))
                else:
                    measurements.append(float(np.mean(non_white)))

            if not measurements:
                self.root.after(100, self._auto_exposure_step)
                return

            metered = sum(measurements) / len(measurements)
            error = metered - s["target"]
            print(
                f"Auto-exposure [{s['iteration']}]: masked_mean={metered:.1f}, target={s['target']}, exposure={self.camera_exposure}, error={error:.1f}")

            if abs(error) <= s["tolerance"] or s["iteration"] >= s["max_iter"]:
                print(
                    f"Auto-exposure converged at exposure={self.camera_exposure} (masked_mean={metered:.1f})")
                self._auto_exposure_done()
                return

            # Linear proportional adjustment: new_exposure = exposure * (target / metered)
            # Dampen to avoid overshoot: blend 60% towards ideal, keep 40% current
            if metered > 0:
                ideal_exposure = self.camera_exposure * (s["target"] / metered)
                new_exposure = int(self.camera_exposure *
                                   0.4 + ideal_exposure * 0.6)
            else:
                new_exposure = self.camera_exposure + 50

            # Clamp to reasonable range
            new_exposure = max(1, min(new_exposure, 10000))

            self.camera_exposure = new_exposure
            s["sensor"].set_option(rs.option.exposure, self.camera_exposure)
            s["iteration"] += 1
            s["prev_error"] = error
            s["flush_count"] = 0  # Reset flush for next iteration

            self.root.after(50, self._auto_exposure_step)

        except Exception as e:
            print(f"Auto-exposure error: {e}")
            self._auto_exposure_done()

    def _adjust_camera(self, param, delta):
        """Adjust camera exposure or contrast via keyboard shortcut."""
        if not self.pipeline or not self.camera_running:
            return
        try:
            sensor = self.pipeline.get_active_profile(
            ).get_device().query_sensors()[1]
            if param == "exposure":
                self.camera_exposure = max(1, self.camera_exposure + delta)
                sensor.set_option(rs.option.exposure, self.camera_exposure)
            elif param == "contrast":
                self.camera_contrast = max(
                    0, min(100, self.camera_contrast + delta))
                sensor.set_option(rs.option.contrast, self.camera_contrast)
            elif param == "brightness_target":
                self.brightness_target = max(
                    10, min(245, self.brightness_target + delta))
            print(f"Camera {param} set to {getattr(self, f'camera_{param}')
                  if param != 'brightness_target' else self.brightness_target}")
            self._update_camera_info_label()
        except Exception as e:
            print(f"Failed to adjust {param}: {e}")

    def _update_camera_info_label(self):
        """Update the on-screen camera info text."""
        if hasattr(self, 'camera_canvas'):
            self.camera_canvas.delete("camera_info")
            self.camera_canvas.create_text(
                10, 10, anchor=tk.NW,
                text=f"Exp: {self.camera_exposure}  |  Con: {self.camera_contrast}  |  BrTgt: {self.brightness_target}   [E/D] exp  [R/F] con  [T/G] target",
                fill="#00f5d4", font=("Consolas", 10),
                tags="camera_info"
            )

    def _create_widgets(self):
        """Create all GUI widgets with vibrant styling."""
        # Configure ttk styles with vibrant colors
        style = ttk.Style()
        style.theme_use('clam')

        # Configure vibrant styles
        style.configure("Vibrant.TFrame", background=self.PALETTE["bg_dark"])
        style.configure("Title.TLabel", font=("Arial", 18, "bold"),
                        foreground=self.PALETTE["accent_cyan"], background=self.PALETTE["bg_dark"])
        style.configure("Step.TLabel", font=("Arial", 12, "bold"),
                        foreground=self.PALETTE["accent_pink"], background=self.PALETTE["bg_dark"])
        style.configure("Vibrant.TLabelframe",
                        background=self.PALETTE["bg_dark"])
        style.configure("Vibrant.TLabelframe.Label", font=("Arial", 11, "bold"),
                        foreground=self.PALETTE["accent_yellow"], background=self.PALETTE["bg_dark"])
        style.configure("Action.TButton", font=(
            "Arial", 11, "bold"), padding=8)
        style.map("Action.TButton",
                  foreground=[('active', self.PALETTE["bg_dark"]),
                              ('!active', self.PALETTE["text_bright"])],
                  background=[('active', self.PALETTE["accent_cyan"]), ('!active', self.PALETTE["accent_purple"])])

        # Main background canvas for gradient and particles
        self.bg_canvas = tk.Canvas(
            self.root, width=1750, height=820, highlightthickness=0)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)
        self._draw_gradient_background()
        self._init_particles()

        # Main container with two columns
        main_frame = tk.Frame(self.bg_canvas, bg=self.PALETTE["bg_dark"])
        self.bg_canvas.create_window(
            875, 410, window=main_frame, width=1730, height=800)

        # Left side - Camera preview with glowing border
        camera_container = tk.Frame(main_frame, bg=self.PALETTE["bg_dark"])
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=True, padx=(10, 10), pady=10)

        # Glowing camera frame label
        # Camera header row with label and auto exposure button
        camera_header = tk.Frame(camera_container, bg=self.PALETTE["bg_dark"])
        camera_header.pack(fill=tk.X, pady=(0, 5))

        camera_label = tk.Label(camera_header, text="📷 LIVE CAMERA FEED",
                                font=("Arial", 12, "bold"), fg=self.PALETTE["accent_cyan"],
                                bg=self.PALETTE["bg_dark"])
        camera_label.pack(side=tk.LEFT, expand=True)

        self.auto_exp_btn = tk.Button(camera_header, text="☀ AUTO EXP", command=self._auto_exposure,
                                      font=("Arial", 9, "bold"), fg=self.PALETTE["bg_dark"],
                                      bg=self.PALETTE["accent_yellow"], activebackground=self.PALETTE["accent_orange"],
                                      activeforeground=self.PALETTE["bg_dark"], relief=tk.FLAT, padx=8, pady=3,
                                      cursor="hand2")
        self.auto_exp_btn.pack(side=tk.RIGHT)

        # Camera canvas with decorative border
        self.camera_frame_canvas = tk.Canvas(camera_container, width=1300, height=740,
                                             bg=self.PALETTE["bg_mid"], highlightthickness=0)
        self.camera_frame_canvas.pack()

        # Draw glowing border
        self._draw_glow_border(self.camera_frame_canvas, 1300, 740)

        # Actual camera canvas inside
        self.camera_canvas = tk.Canvas(self.camera_frame_canvas, width=1280, height=720,
                                       bg=self.PALETTE["bg_light"], highlightthickness=0)
        self.camera_frame_canvas.create_window(
            650, 370, window=self.camera_canvas)
        self.camera_image_id = None

        # Bind click event for target selection
        self.camera_canvas.bind("<Button-1>", self._on_camera_click)

        # Animated placeholder
        self._draw_camera_placeholder()

        # Right side - Controls with styled background
        self.controls_frame = tk.Frame(
            main_frame, bg=self.PALETTE["bg_dark"], width=380)
        self.controls_frame.pack(
            side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        self.controls_frame.pack_propagate(False)

        # Animated title header
        title_frame = tk.Frame(self.controls_frame, bg=self.PALETTE["bg_dark"])
        title_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = tk.Label(title_frame, text="🤖 VLM BLOCK CONTROL 🎮",
                                    font=("Arial", 16, "bold"), fg=self.PALETTE["accent_cyan"],
                                    bg=self.PALETTE["bg_dark"])
        self.title_label.pack()

        # Subtitle with animation
        self.subtitle_label = tk.Label(title_frame, text="✨ Interactive Robot Programming ✨",
                                       font=("Arial", 10), fg=self.PALETTE["accent_pink"],
                                       bg=self.PALETTE["bg_dark"])
        self.subtitle_label.pack()

        # Step indicator with style
        step_frame = tk.Frame(self.controls_frame,
                              bg=self.PALETTE["bg_mid"], padx=15, pady=8)
        step_frame.pack(fill=tk.X, pady=(0, 10))

        self.step_label = tk.Label(step_frame, text="⚡ Step 1 of 3",
                                   font=("Arial", 12, "bold"), fg=self.PALETTE["accent_yellow"],
                                   bg=self.PALETTE["bg_mid"])
        self.step_label.pack()

        # Custom progress bar canvas
        self.progress_canvas = tk.Canvas(self.controls_frame, width=340, height=20,
                                         bg=self.PALETTE["bg_mid"], highlightthickness=0)
        self.progress_canvas.pack(pady=(0, 15))
        self._draw_progress_bar(33)

        # Container for step content
        self.step_container = tk.Frame(
            self.controls_frame, bg=self.PALETTE["bg_dark"])
        self.step_container.pack(fill=tk.BOTH, expand=True)

        # Create all step frames
        self._create_step1_frame()
        self._create_step2_stack_frame()
        self._create_step2_sort_frame()
        self._create_step3_stack_frame()

        # Navigation buttons with vibrant styling
        nav_frame = tk.Frame(self.controls_frame, bg=self.PALETTE["bg_dark"])
        nav_frame.pack(fill=tk.X, pady=15)

        # Custom styled buttons using tk.Button for more control
        self.back_btn = tk.Button(nav_frame, text="◀ BACK", command=self._go_back,
                                  font=("Arial", 10, "bold"), fg=self.PALETTE["text_bright"],
                                  bg=self.PALETTE["accent_purple"], activebackground=self.PALETTE["accent_pink"],
                                  activeforeground="white", relief=tk.FLAT, padx=15, pady=8,
                                  cursor="hand2")
        self.back_btn.pack(side=tk.LEFT)

        self.next_btn = tk.Button(nav_frame, text="CONFIRM ▶", command=self._go_next,
                                  font=("Arial", 10, "bold"), fg=self.PALETTE["bg_dark"],
                                  bg=self.PALETTE["accent_cyan"], activebackground=self.PALETTE["accent_yellow"],
                                  activeforeground=self.PALETTE["bg_dark"], relief=tk.FLAT, padx=15, pady=8,
                                  cursor="hand2")
        self.next_btn.pack(side=tk.RIGHT)

        self.confirm_btn = tk.Button(nav_frame, text="✓ FINISH", command=self._confirm,
                                     font=("Arial", 10, "bold"), fg=self.PALETTE["bg_dark"],
                                     bg=self.PALETTE["accent_cyan"], activebackground=self.PALETTE["accent_yellow"],
                                     activeforeground=self.PALETTE["bg_dark"], relief=tk.FLAT, padx=15, pady=8,
                                     cursor="hand2")
        self.confirm_btn.pack(side=tk.RIGHT, padx=5)

        cancel_btn = tk.Button(nav_frame, text="✕ CANCEL", command=self._cancel,
                               font=("Arial", 10, "bold"), fg=self.PALETTE["text_dim"],
                               bg=self.PALETTE["bg_mid"], activebackground="#ff6b6b",
                               activeforeground="white", relief=tk.FLAT, padx=15, pady=8,
                               cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        # Start animations
        self._animate_title()
        self._animate_particles()
        self._animate_glow()

    def _draw_gradient_background(self):
        """Draw a vibrant gradient background."""
        width, height = 1750, 820
        # Create gradient from dark purple to dark blue with some accent
        for i in range(height):
            ratio = i / height
            # Interpolate between colors
            r = int(13 + ratio * 15)  # Dark purple to slightly lighter
            g = int(2 + ratio * 8)
            b = int(33 + ratio * 40)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.bg_canvas.create_line(0, i, width, i, fill=color)

        # Add some decorative geometric shapes
        for _ in range(8):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(50, 150)
            alpha_hex = random.choice(['15', '20', '25', '30'])
            self.bg_canvas.create_oval(
                x - size, y - size, x + size, y + size,
                outline=f"#{alpha_hex}0050", width=2
            )

        # Add diagonal accent lines
        for i in range(0, 1400, 100):
            self.bg_canvas.create_line(
                i, 0, i - 400, 820,
                fill="#1a0a30", width=1
            )

    def _init_particles(self):
        """Initialize floating particle effects."""
        self.particles = []
        for _ in range(30):
            particle = {
                'x': random.randint(0, 1750),
                'y': random.randint(0, 820),
                'size': random.randint(2, 6),
                'speed': random.uniform(0.3, 1.2),
                'color': random.choice(['#9d4edd', '#e91e8c', '#00f5d4', '#ffd23f']),
                'id': None
            }
            particle['id'] = self.bg_canvas.create_oval(
                particle['x'] - particle['size'], particle['y'] -
                particle['size'],
                particle['x'] + particle['size'], particle['y'] +
                particle['size'],
                fill=particle['color'], outline=''
            )
            self.particles.append(particle)

    def _animate_particles(self):
        """Animate floating particles."""
        if not self.animation_running:
            return

        for p in self.particles:
            p['y'] -= p['speed']
            if p['y'] < -10:
                p['y'] = 830
                p['x'] = random.randint(0, 1750)

            self.bg_canvas.coords(
                p['id'],
                p['x'] - p['size'], p['y'] - p['size'],
                p['x'] + p['size'], p['y'] + p['size']
            )

        self.root.after(50, self._animate_particles)

    def _draw_glow_border(self, canvas, width, height):
        """Draw a glowing border effect."""
        # Multiple layers for glow effect
        colors = ['#3d1a78', '#5a2d9e', '#7b2cbf', '#9d4edd']
        for i, color in enumerate(colors):
            offset = 8 - i * 2
            canvas.create_rectangle(
                offset, offset, width - offset, height - offset,
                outline=color, width=2
            )

        # Corner accents
        accent_size = 15
        for x, y in [(0, 0), (width, 0), (0, height), (width, height)]:
            canvas.create_rectangle(
                x - accent_size if x > 0 else x,
                y - accent_size if y > 0 else y,
                x + accent_size if x == 0 else x,
                y + accent_size if y == 0 else y,
                fill=self.PALETTE['accent_cyan'], outline=''
            )

    def _animate_glow(self):
        """Animate the glowing effects."""
        if not self.animation_running:
            return

        self.glow_phase = (self.glow_phase + 0.1) % (2 * math.pi)

        # Pulse the title color
        intensity = int(128 + 127 * math.sin(self.glow_phase))
        color = f'#{intensity:02x}f5d4'
        try:
            self.title_label.config(fg=color)
        except:
            pass

        self.root.after(100, self._animate_glow)

    def _animate_title(self):
        """Animate title with rainbow effect."""
        if not self.animation_running:
            return

        colors = ['#00f5d4', '#e91e8c', '#ffd23f', '#7b2cbf', '#ff6b35']
        current_color = colors[int(self.glow_phase * 2) % len(colors)]
        try:
            self.subtitle_label.config(fg=current_color)
        except:
            pass

        self.root.after(500, self._animate_title)

    def _draw_camera_placeholder(self):
        """Draw an animated placeholder for camera."""
        cx, cy = 640, 360

        # Draw cyberpunk-style grid
        for i in range(0, 1280, 40):
            self.camera_canvas.create_line(
                i, 0, i, 720, fill='#2a1a4a', width=1, tags='placeholder')
        for i in range(0, 720, 40):
            self.camera_canvas.create_line(
                0, i, 1280, i, fill='#2a1a4a', width=1, tags='placeholder')

        # Central icon
        self.camera_canvas.create_oval(cx-50, cy-50, cx+50, cy+50,
                                       outline=self.PALETTE['accent_purple'], width=3, tags='placeholder')
        self.camera_canvas.create_oval(cx-30, cy-30, cx+30, cy+30,
                                       outline=self.PALETTE['accent_cyan'], width=2, tags='placeholder')
        self.camera_canvas.create_text(cx, cy, text="📷", font=("Arial", 24),
                                       fill=self.PALETTE['accent_cyan'], tags='placeholder')
        self.camera_canvas.create_text(cx, cy + 80, text="Initializing Camera...",
                                       font=("Arial", 14, "bold"), fill=self.PALETTE['accent_pink'], tags='placeholder')
        self.camera_canvas.create_text(cx, cy + 105, text="Click here to select target position",
                                       font=("Arial", 10), fill=self.PALETTE['text_dim'], tags='placeholder')

    def _draw_progress_bar(self, value):
        """Draw a custom vibrant progress bar."""
        self.progress_canvas.delete('all')

        # Background
        self.progress_canvas.create_rectangle(
            0, 0, 340, 20, fill=self.PALETTE['bg_light'], outline='')

        # Progress fill with gradient effect
        width = int(340 * value / 100)
        if width > 0:
            # Create gradient segments
            colors = ['#7b2cbf', '#9d4edd', '#e91e8c', '#ff6b35']
            segment_width = width // len(colors) + 1
            for i, color in enumerate(colors):
                x1 = i * segment_width
                x2 = min((i + 1) * segment_width, width)
                if x1 < width:
                    self.progress_canvas.create_rectangle(
                        x1, 2, x2, 18, fill=color, outline='')

        # Glowing edge
        if width > 0:
            self.progress_canvas.create_rectangle(width - 3, 0, width, 20,
                                                  fill=self.PALETTE['accent_cyan'], outline='')

        # Border
        self.progress_canvas.create_rectangle(
            0, 0, 340, 20, outline=self.PALETTE['accent_purple'], width=2)

    def _create_step1_frame(self):
        """Create Step 1: Task Selection with vibrant visual cards."""
        frame = tk.Frame(self.step_container, bg=self.PALETTE["bg_dark"])

        # Canvas for visual task selection
        canvas = tk.Canvas(frame, width=340, height=380,
                           bg=self.PALETTE["bg_dark"], highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Decorative background pattern
        for i in range(0, 380, 30):
            canvas.create_line(
                0, i, 340, i, fill=self.PALETTE["bg_mid"], width=1)

        # Title with glow effect
        canvas.create_text(172, 27, text="🎮 SELECT YOUR MISSION",
                           fill=self.PALETTE["bg_light"], font=("Arial", 14, "bold"))
        canvas.create_text(170, 25, text="🎮 SELECT YOUR MISSION",
                           fill=self.PALETTE["accent_cyan"], font=("Arial", 14, "bold"))
        canvas.create_text(170, 50, text="⚡ Click a card to begin ⚡",
                           fill=self.PALETTE["accent_yellow"], font=("Arial", 10))

        # Task card dimensions
        card_width = 300
        card_height = 130

        # Stack card with neon effect
        stack_y = 75
        stack_selected = self.task_var.get() == "stack"
        # Glow layer
        canvas.create_rectangle(
            17, stack_y - 3, 23 + card_width, stack_y + card_height + 3,
            fill="", outline=self.PALETTE["accent_purple"], width=2
        )
        self.stack_card = canvas.create_rectangle(
            20, stack_y, 20 + card_width, stack_y + card_height,
            fill=self.PALETTE["bg_mid"] if stack_selected else self.PALETTE["bg_light"],
            outline=self.PALETTE["accent_purple"], width=3
        )
        # Icon background
        canvas.create_oval(50, stack_y + 30, 90, stack_y +
                           70, fill=self.PALETTE["accent_orange"], outline="")
        canvas.create_text(70, stack_y + 50, text="📦", font=("Arial", 20))
        canvas.create_text(190, stack_y + 40, text="STACK BLOCKS",
                           fill=self.PALETTE["text_bright"], font=("Arial", 14, "bold"))
        canvas.create_text(190, stack_y + 65, text="Build a colorful tower",
                           fill=self.PALETTE["text_dim"], font=("Arial", 10))
        canvas.create_text(190, stack_y + 85, text="by stacking blocks up",
                           fill=self.PALETTE["text_dim"], font=("Arial", 10))
        canvas.create_text(190, stack_y + 110, text="▶ CLICK TO SELECT",
                           fill=self.PALETTE["accent_cyan"], font=("Arial", 9, "bold"))

        # Bind click events - auto advance to step 2 with visual feedback
        def select_stack(event):
            self.task_var.set("stack")
            canvas.itemconfig(
                self.stack_card, fill=self.PALETTE["accent_purple"], outline=self.PALETTE["accent_cyan"])
            # Flash effect then advance
            self.root.after(300, lambda: self._show_step(2))

        # Bind to all elements in card region
        canvas.tag_bind(self.stack_card, "<Button-1>", select_stack)

        # Create invisible clickable region over the card
        stack_region = canvas.create_rectangle(
            20, stack_y, 20 + card_width, stack_y + card_height, fill="", outline="")
        canvas.tag_bind(stack_region, "<Button-1>", select_stack)

        self.step_frames["step1"] = frame

    def _create_step2_stack_frame(self):
        """Create Step 2 for Stack: Drag-and-drop stacking order."""
        frame = tk.Frame(self.step_container, bg=self.PALETTE["bg_dark"])

        # Header
        header = tk.Label(frame, text="🏗️ ARRANGE STACK ORDER",
                          font=("Arial", 12, "bold"), fg=self.PALETTE["accent_yellow"],
                          bg=self.PALETTE["bg_dark"])
        header.pack(pady=(5, 5))

        # Create stack builder
        self.stack_builder = StackBuilder(frame, width=340, height=430)
        self.stack_builder.pack(fill=tk.BOTH, expand=True)

        self.step_frames["step2_stack"] = frame

    def _create_step2_sort_frame(self):
        """Create Step 2 for Sort: Click on camera to map each block's target."""
        frame = tk.Frame(self.step_container, bg=self.PALETTE["bg_dark"])

        # Canvas for block selection UI
        canvas = tk.Canvas(frame, width=340, height=480,
                           bg=self.PALETTE["bg_dark"], highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.sort_step2_canvas = canvas

        # Draw decorative background pattern
        for i in range(0, 480, 25):
            canvas.create_line(
                0, i, 340, i, fill=self.PALETTE["bg_mid"], width=1)
        for i in range(0, 340, 25):
            canvas.create_line(
                i, 0, i, 480, fill=self.PALETTE["bg_mid"], width=1)

        # Title with glow
        canvas.create_text(172, 22, text="🎯 MAP BLOCKS TO TARGETS",
                           fill=self.PALETTE["bg_light"], font=("Arial", 12, "bold"))
        canvas.create_text(170, 20, text="🎯 MAP BLOCKS TO TARGETS",
                           fill=self.PALETTE["accent_pink"], font=("Arial", 12, "bold"))
        canvas.create_text(170, 42, text="1️⃣ Click a block to select",
                           fill=self.PALETTE["accent_cyan"], font=("Arial", 9))
        canvas.create_text(170, 58, text="2️⃣ Click on camera to place it",
                           fill=self.PALETTE["accent_cyan"], font=("Arial", 9))

        # Block selection buttons - 5 blocks
        block_colors = {
            "red": {"fill": "#E74C3C", "hover": "#FF6B5B"},
            "yellow": {"fill": "#F1C40F", "hover": "#F4D03F"},
            "light blue": {"fill": "#5DADE2", "hover": "#85C1E9"},
            "dark blue": {"fill": "#2471A3", "hover": "#2E86C1"},
            "green": {"fill": "#27AE60", "hover": "#2ECC71"},
        }
        display_labels = {
            "red": "RED", "yellow": "YELLOW", "light blue": "LT BLUE",
            "dark blue": "DK BLUE", "green": "GREEN",
        }

        y_start = 78
        block_height = 55

        for i, (color, colors) in enumerate(block_colors.items()):
            y = y_start + i * (block_height + 8)

            # Shadow
            canvas.create_rectangle(
                24, y + 3, 134, y + block_height + 3, fill="#0f0f23", outline="", tags=f"block_{color}")

            # Main block button
            rect = canvas.create_rectangle(
                20, y, 130, y + block_height,
                fill=colors["fill"], outline="white", width=3, tags=f"block_{color}"
            )

            # Block label
            text_color = "#1a1a2e" if color == "yellow" else "white"
            canvas.create_text(75, y + 20, text=display_labels[color], fill=text_color, font=(
                "Arial", 10, "bold"), tags=f"block_{color}")
            canvas.create_text(75, y + 39, text="Click to select",
                               fill="#cccccc", font=("Arial", 7), tags=f"block_{color}")

            # Status indicator area (right side) with neon styling
            status_rect = canvas.create_rectangle(
                145, y + 8, 320, y + block_height - 8,
                fill=self.PALETTE["bg_mid"], outline=self.PALETTE["accent_purple"], width=2, tags=f"status_{color}"
            )
            status_text = canvas.create_text(
                232, y + block_height // 2, text="⚠️ Not set",
                fill="#ff6b6b", font=("Arial", 9, "bold"), tags=f"status_text_{color}"
            )

            # Store references
            self.sort_block_buttons[color] = {
                "rect": rect, "status_rect": status_rect, "status_text": status_text,
                "y": y, "colors": colors
            }

            # Bind click events
            def make_select_handler(c):
                return lambda event: self._select_sort_block(c)

            canvas.tag_bind(f"block_{color}",
                            "<Button-1>", make_select_handler(color))
            canvas.tag_bind(
                f"block_{color}", "<Enter>", lambda e, c=color: self._on_block_hover(c, True))
            canvas.tag_bind(
                f"block_{color}", "<Leave>", lambda e, c=color: self._on_block_hover(c, False))

        # Currently selected indicator with vibrant styling
        self.sort_selection_text = canvas.create_text(
            170, 465, text="👆 Select a block to map...",
            fill=self.PALETTE["accent_yellow"], font=("Arial", 11, "bold")
        )

        self.step_frames["step2_sort"] = frame

    def _select_sort_block(self, color):
        """Select a block to map to camera position."""
        self.current_mapping_block = color

        # Update visual selection with neon glow
        for c, btn in self.sort_block_buttons.items():
            if c == color:
                self.sort_step2_canvas.itemconfig(
                    btn["rect"], outline=self.PALETTE["accent_cyan"], width=5)
            else:
                self.sort_step2_canvas.itemconfig(
                    btn["rect"], outline="white", width=3)

        # Update instruction text with vibrant color
        self.sort_step2_canvas.itemconfig(
            self.sort_selection_text,
            text=f"🎯 Click on camera to place {color.upper()}!",
            fill=self.PALETTE["accent_cyan"]
        )

    def _on_block_hover(self, color, entering):
        """Handle hover effect on sort block buttons."""
        if color in self.sort_block_buttons:
            btn = self.sort_block_buttons[color]
            if entering and self.current_mapping_block != color:
                self.sort_step2_canvas.itemconfig(
                    btn["rect"], fill=btn["colors"]["hover"])
                self.sort_step2_canvas.config(cursor="hand2")
            else:
                self.sort_step2_canvas.itemconfig(
                    btn["rect"], fill=btn["colors"]["fill"])
                self.sort_step2_canvas.config(cursor="")

    def _update_sort_block_status(self, color, position):
        """Update the status display for a mapped block."""
        if color in self.sort_block_buttons:
            btn = self.sort_block_buttons[color]
            self.sort_step2_canvas.itemconfig(
                btn["status_text"],
                text=f"✅ ({position[0]}, {position[1]})",
                fill=self.PALETTE["accent_cyan"]
            )

    def _create_step3_stack_frame(self):
        """Create Step 3 for Stack: Select a detected white paper as target."""
        frame = tk.Frame(self.step_container, bg=self.PALETTE["bg_dark"])

        # Canvas for instructions
        canvas = tk.Canvas(frame, width=340, height=380,
                           bg=self.PALETTE["bg_dark"], highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Draw decorative grid background
        for i in range(0, 380, 30):
            canvas.create_line(
                0, i, 340, i, fill=self.PALETTE["bg_mid"], width=1)
        for i in range(0, 340, 30):
            canvas.create_line(
                i, 0, i, 380, fill=self.PALETTE["bg_mid"], width=1)

        # Title with glow effect
        canvas.create_text(172, 32, text="📍 SELECT TARGET PAPER",
                           fill=self.PALETTE["bg_light"], font=("Arial", 13, "bold"))
        canvas.create_text(170, 30, text="📍 SELECT TARGET PAPER",
                           fill=self.PALETTE["accent_orange"], font=("Arial", 13, "bold"))
        canvas.create_text(170, 58, text="Click a highlighted white paper",
                           fill=self.PALETTE["accent_cyan"], font=("Arial", 10))

        # Animated instruction box with neon border
        canvas.create_rectangle(
            25, 85, 315, 230, fill=self.PALETTE["bg_mid"], outline=self.PALETTE["accent_purple"], width=3)
        canvas.create_rectangle(28, 88, 312, 227, fill="",
                                outline=self.PALETTE["accent_pink"], width=1)

        # Instructions with icons
        canvas.create_text(170, 110, text="📄 White papers are auto-detected",
                           fill=self.PALETTE["text_bright"], font=("Arial", 10))
        canvas.create_text(170, 135, text="and highlighted on the camera.",
                           fill=self.PALETTE["text_bright"], font=("Arial", 10))
        canvas.create_text(170, 165, text="👆 Click on a highlighted paper",
                           fill=self.PALETTE["text_bright"], font=("Arial", 10))
        canvas.create_text(170, 190, text="to set it as the target area.",
                           fill=self.PALETTE["text_bright"], font=("Arial", 10))
        canvas.create_text(170, 218, text="🎯 The center will be the drop point!",
                           fill=self.PALETTE["accent_cyan"], font=("Arial", 10, "bold"))

        # Status indicator
        status_bg = canvas.create_rectangle(
            50, 250, 290, 295, fill=self.PALETTE["bg_light"], outline=self.PALETTE["accent_pink"], width=2)
        self.target_status_text = canvas.create_text(
            170, 272, text="⚠️ No target selected yet",
            fill="#ff6b6b", font=("Arial", 12, "bold")
        )
        self.step3_canvas = canvas

        # Paper detection status
        self.paper_detect_status = canvas.create_text(
            170, 325, text="🔍 Searching for white papers...",
            fill=self.PALETTE["accent_yellow"], font=("Arial", 10, "bold")
        )
        canvas.create_text(170, 355, text="✨ Then click Finish to complete! ✨",
                           fill=self.PALETTE["text_dim"], font=("Arial", 9))

        self.step_frames["step3_stack"] = frame

    def _detect_white_papers(self, frame_bgr):
        """Detect white paper regions in the camera frame using minAreaRect.

        Returns list of ((cx, cy), (w, h), angle) tuples from cv2.minAreaRect.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        papers = []
        min_area = 50000  # Minimum area to count as a paper
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            papers.append(rect)

        # Sort by area descending
        papers.sort(key=lambda r: r[1][0] * r[1][1], reverse=True)
        return papers

    def _on_camera_click(self, event):
        """Handle click on camera image to select target location."""
        task = self.task_var.get()

        # Stack mode: Step 3 - select a detected white paper
        if task == "stack" and self.current_step == 3:
            x, y = event.x, event.y

            # Find which detected paper was clicked
            clicked_paper = None
            for idx, rect in enumerate(self.detected_papers):
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                # Point-in-polygon test
                if cv2.pointPolygonTest(box, (float(x), float(y)), False) >= 0:
                    clicked_paper = idx
                    break

            if clicked_paper is None:
                # Clicked outside any detected paper
                return

            rect = self.detected_papers[clicked_paper]
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])

            x_norm = int((center_x / 1280) * 1000)
            y_norm = int((center_y / 720) * 1000)

            self.target_click_pos = (y_norm, x_norm)
            self.selected_paper_idx = clicked_paper

            # Remove old marker
            self.camera_canvas.delete("target_marker")

            # Draw marker at center of selected paper
            self._draw_camera_marker(
                center_x, center_y, "#58D68D", "target_marker")
            self.target_marker = True

            if hasattr(self, 'step3_canvas'):
                self.step3_canvas.itemconfig(
                    self.target_status_text,
                    text=f"✅ Paper #{clicked_paper + 1} selected",
                    fill="#58D68D"
                )

        # Sort mode: Step 2 for mapping each block
        elif task == "sort" and self.current_step == 2:
            if not self.current_mapping_block:
                return  # No block selected

            x, y = event.x, event.y
            x_norm = int((x / 1280) * 1000)
            y_norm = int((y / 720) * 1000)

            color = self.current_mapping_block
            self.sort_block_positions[color] = (y_norm, x_norm)

            # Remove old marker for this color
            self.camera_canvas.delete(f"sort_marker_{color}")

            # Draw colored marker
            marker_colors = {
                "red": "#E74C3C", "green": "#27AE60", "dark blue": "#2471A3",
                "yellow": "#F1C40F", "light blue": "#5DADE2",
            }
            display_initials = {
                "red": "R", "green": "G", "dark blue": "D",
                "yellow": "Y", "light blue": "L",
            }
            self._draw_camera_marker(x, y, marker_colors.get(color, "#ffffff"),
                                     f"sort_marker_{color}", label=display_initials.get(color, "?"))

            # Update status in the UI
            self._update_sort_block_status(color, (x_norm, y_norm))

            # Clear selection and update text
            self.current_mapping_block = None
            for c, btn in self.sort_block_buttons.items():
                self.sort_step2_canvas.itemconfig(
                    btn["rect"], outline="white", width=3)

            # Check if all blocks are mapped
            mapped_count = len(self.sort_block_positions)
            total_blocks = len(self.BLOCK_COLORS)
            if mapped_count == total_blocks:
                self.sort_step2_canvas.itemconfig(
                    self.sort_selection_text,
                    text="🎉 All blocks mapped! Click FINISH!",
                    fill=self.PALETTE["accent_cyan"]
                )
            else:
                self.sort_step2_canvas.itemconfig(
                    self.sort_selection_text,
                    text=f"✨ Mapped {mapped_count}/{total_blocks} - Select next!",
                    fill=self.PALETTE["accent_yellow"]
                )

    def _draw_camera_marker(self, x, y, color, tag, label=None):
        """Draw a marker on the camera canvas."""
        # Outer circle
        self.camera_canvas.create_oval(
            x - 20, y - 20, x + 20, y + 20,
            outline=color, width=3, tags=tag
        )
        # Inner circle
        self.camera_canvas.create_oval(
            x - 8, y - 8, x + 8, y + 8,
            fill=color, outline="white", width=2, tags=tag
        )
        # Crosshair lines
        self.camera_canvas.create_line(
            x - 30, y, x - 22, y, fill=color, width=2, tags=tag)
        self.camera_canvas.create_line(
            x + 22, y, x + 30, y, fill=color, width=2, tags=tag)
        self.camera_canvas.create_line(
            x, y - 30, x, y - 22, fill=color, width=2, tags=tag)
        self.camera_canvas.create_line(
            x, y + 22, x, y + 30, fill=color, width=2, tags=tag)

        # Optional label
        if label:
            self.camera_canvas.create_text(
                x, y, text=label, fill="white", font=("Arial", 10, "bold"), tags=tag
            )

    def _show_step(self, step):
        """Show the specified step and hide others."""
        self.current_step = step

        # Hide all step frames
        for frame in self.step_frames.values():
            frame.pack_forget()

        # Reset all target points when returning to step 1
        if step == 1:
            # Clear stack mode target
            self.target_click_pos = None
            self.target_marker = None
            self.selected_paper_idx = None
            self.detected_papers = []
            self.camera_canvas.delete("target_marker")

            # Reset paper detection status
            if hasattr(self, 'paper_detect_status'):
                self.step3_canvas.itemconfig(
                    self.paper_detect_status,
                    text="\ud83d\udd0d Searching for white papers...",
                    fill=self.PALETTE["accent_yellow"]
                )

            # Clear sort mode targets
            self.sort_block_positions = {}
            self.current_mapping_block = None
            for color in self.BLOCK_COLORS:
                self.camera_canvas.delete(f"sort_marker_{color}")
                # Reset status texts in sort step 2
                if color in self.sort_block_buttons:
                    btn = self.sort_block_buttons[color]
                    self.sort_step2_canvas.itemconfig(
                        btn["status_text"],
                        text="⚠️ Not set",
                        fill="#ff6b6b"
                    )
                    self.sort_step2_canvas.itemconfig(
                        btn["rect"], outline="white", width=3)

            # Reset sort selection text
            if hasattr(self, 'sort_selection_text'):
                self.sort_step2_canvas.itemconfig(
                    self.sort_selection_text,
                    text="👆 Select a block to map...",
                    fill=self.PALETTE["accent_yellow"]
                )

            # Reset stack step 3 status
            if hasattr(self, 'step3_canvas') and hasattr(self, 'target_status_text'):
                self.step3_canvas.itemconfig(
                    self.target_status_text,
                    text="⚠️ No target selected yet",
                    fill="#ff6b6b"
                )

        task = self.task_var.get()

        # Stack has 3 steps, Sort has 2 steps
        if task == "stack":
            self.total_steps = 3
        else:
            self.total_steps = 2

        if task == "stack":
            if step == 1:
                self.step_frames["step1"].pack(fill=tk.BOTH, expand=True)
            elif step == 2:
                self.step_frames["step2_stack"].pack(fill=tk.BOTH, expand=True)
            elif step == 3:
                self.step_frames["step3_stack"].pack(fill=tk.BOTH, expand=True)
                # Reset state on entering step 3
                self.selected_paper_idx = None
                if hasattr(self, 'step3_canvas') and not self.target_click_pos:
                    self.step3_canvas.itemconfig(
                        self.target_status_text,
                        text="⚠️ No target selected",
                        fill="#E74C3C"
                    )
                if hasattr(self, 'paper_detect_status'):
                    self.step3_canvas.itemconfig(
                        self.paper_detect_status,
                        text="🔍 Searching for white papers...",
                        fill=self.PALETTE["accent_yellow"]
                    )
        else:  # sort - only 2 steps
            if step == 1:
                self.step_frames["step1"].pack(fill=tk.BOTH, expand=True)
            elif step == 2:
                self.step_frames["step2_sort"].pack(fill=tk.BOTH, expand=True)

        # Update step label and progress bar
        self.step_label.config(text=f"⚡ Step {step} of {self.total_steps}")
        self._draw_progress_bar((step / self.total_steps) * 100)

        # Update navigation buttons based on step and task
        self.back_btn.config(state=tk.NORMAL if step > 1 else tk.DISABLED)

        if step == 1:
            # Step 1: No next button (auto-advance on click)
            self.next_btn.pack_forget()
            self.confirm_btn.pack_forget()
        elif task == "stack" and step == 2:
            # Stack Step 2: Show "Confirm Order" button
            self.confirm_btn.pack_forget()
            self.next_btn.config(text="CONFIRM ORDER ▶")
            self.next_btn.pack(side=tk.RIGHT)
        elif task == "stack" and step == 3:
            # Stack Step 3: Show "Finish" button
            self.next_btn.pack_forget()
            self.confirm_btn.pack(side=tk.RIGHT, padx=5)
        elif task == "sort" and step == 2:
            # Sort Step 2: Show "Finish" button (final step)
            self.next_btn.pack_forget()
            self.confirm_btn.pack(side=tk.RIGHT, padx=5)

    def _go_back(self):
        """Go to previous step."""
        if self.current_step > 1:
            self._show_step(self.current_step - 1)

    def _go_next(self):
        """Go to next step."""
        # Validate minimum 2 blocks in stack step 2
        if self.task_var.get() == "stack" and self.current_step == 2:
            stack_order = self.stack_builder.get_stack_order()
            if len(stack_order) < 2:
                import tkinter.messagebox
                tkinter.messagebox.showwarning(
                    "Warning", "Please drag at least 2 blocks into the stack before confirming.")
                return
        if self.current_step < self.total_steps:
            self._show_step(self.current_step + 1)

    def _start_camera_feed(self):
        """Start updating the camera feed."""
        if not self.camera_running or not self.pipeline:
            return
        self._update_camera()

    def _update_camera(self):
        """Update camera preview with new frame."""
        if not self.camera_running or not self.pipeline:
            return

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()

            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                display_image = cv2.resize(color_image, (1280, 720))

                # Detect and highlight white papers when in stack step 3
                if self.task_var.get() == "stack" and self.current_step == 3:
                    # Convert RGB (RealSense) to BGR for OpenCV processing
                    bgr_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                    self.detected_papers = self._detect_white_papers(bgr_image)

                    # Draw highlights on the display image
                    for idx, rect in enumerate(self.detected_papers):
                        # Green highlight for detected papers, cyan for selected
                        if self.selected_paper_idx == idx:
                            # Selected: bright green
                            color_bgr = (93, 211, 158)
                            thickness = 4
                        else:
                            color_bgr = (0, 255, 255)  # Detected: cyan
                            thickness = 3
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(
                            display_image, [box], 0, color_bgr, thickness)
                        # Label at center
                        cx, cy = int(rect[0][0]), int(rect[0][1])
                        label = f"Paper #{idx + 1}"
                        cv2.putText(display_image, label, (cx - 40, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

                    # Update paper detection status in step 3 panel
                    if hasattr(self, 'paper_detect_status'):
                        n = len(self.detected_papers)
                        if n == 0:
                            self.step3_canvas.itemconfig(
                                self.paper_detect_status,
                                text="❌ No white papers detected",
                                fill="#ff6b6b"
                            )
                        else:
                            self.step3_canvas.itemconfig(
                                self.paper_detect_status,
                                text=f"📄 {n} white paper{'s' if n != 1 else ''} detected",
                                fill=self.PALETTE["accent_cyan"]
                            )
                else:
                    self.detected_papers = []
                    self.selected_paper_idx = None

                pil_image = Image.fromarray(display_image)
                photo = ImageTk.PhotoImage(pil_image)

                # Delete placeholder and old image
                self.camera_canvas.delete("placeholder")
                if self.camera_image_id:
                    self.camera_canvas.delete(self.camera_image_id)

                # Create new image (at bottom layer so markers stay on top)
                self.camera_image_id = self.camera_canvas.create_image(
                    0, 0, anchor=tk.NW, image=photo, tags="camera_image")
                # Keep image below markers
                self.camera_canvas.tag_lower("camera_image")
                self.camera_canvas.image = photo  # Keep reference
                self._update_camera_info_label()

        except Exception as e:
            pass

        if self.camera_running:
            self.root.after(33, self._update_camera)

    def _stop_camera(self):
        """Stop the camera pipeline."""
        self.camera_running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None

    def _cancel_all_after(self):
        """Cancel all pending 'after' callbacks to avoid errors on destroy."""
        try:
            # Tk stores pending after callbacks; iterate and cancel them
            for after_id in self.root.tk.call('after', 'info'):
                self.root.after_cancel(after_id)
        except Exception:
            pass

    def _on_close(self):
        """Handle window close event."""
        self.animation_running = False
        self._stop_camera()
        self._cancel_all_after()
        self.result = None
        self.root.quit()
        self.root.destroy()

    def _confirm(self):
        """Store selected values and close the GUI."""
        task = self.task_var.get()

        if task == "stack":
            # Check if target was clicked
            if not self.target_click_pos:
                tk.messagebox.showwarning(
                    "Warning", "Please click on a detected white paper to select a target location.")
                return

            stack_order = self.stack_builder.get_stack_order()

            if len(stack_order) < 2:
                tk.messagebox.showwarning(
                    "Warning", "Please drag at least 2 blocks into the stack.")
                return

            self.animation_running = False
            self._stop_camera()

            self.result = {
                "task": "stack blocks",
                "stack_order": stack_order,
                "objects_to_manipulate": [f"{color} block" for color in stack_order],
                # (y_norm, x_norm) in 0-1000 range
                "target_position": self.target_click_pos,
                "target_location": f"clicked position at normalized coordinates {self.target_click_pos}",
                "brightness_target": self.brightness_target,
                "exposure": self.camera_exposure,
                "contrast": self.camera_contrast,
            }
        else:  # sort
            # Check if at least one block is mapped
            if len(self.sort_block_positions) == 0:
                import tkinter.messagebox
                tkinter.messagebox.showwarning(
                    "Warning", "Please map at least one block to a target position.")
                return

            self.animation_running = False
            self._stop_camera()

            self.result = {
                "task": "sort blocks",
                # {"red": (x, y), ...}
                "block_positions": self.sort_block_positions,
                "objects_to_manipulate": [f"{color} block" for color in self.sort_block_positions.keys()],
                "target_positions": {color: f"normalized coordinates {pos}" for color, pos in self.sort_block_positions.items()},
                "brightness_target": self.brightness_target,
                "exposure": self.camera_exposure,
                "contrast": self.camera_contrast,
            }

        # Save result to YAML file
        self._save_result_yaml()

        self._cancel_all_after()
        self.root.quit()
        self.root.destroy()

    def _save_result_yaml(self):
        """Save the result configuration to a YAML file."""
        if not self.result:
            return

        # Build serializable copy (convert tuples to lists for YAML)
        output = {}
        for key, value in self.result.items():
            if isinstance(value, tuple):
                output[key] = list(value)
            elif isinstance(value, dict):
                output[key] = {k: list(v) if isinstance(
                    v, tuple) else v for k, v in value.items()}
            else:
                output[key] = value

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "VLM_input")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "GUI_output.yaml")
        with open(output_path, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {output_path}")

    def _cancel(self):
        """Close the GUI without storing values."""
        self.animation_running = False
        self._stop_camera()
        self._cancel_all_after()
        self.result = None
        self.root.quit()
        self.root.destroy()

    def get_result(self):
        """Return the selected configuration."""
        return self.result


def get_user_input():
    """
    Display the wizard GUI and return user selection.

    Returns:
        dict or None: Selected configuration.

        For stack task (3 steps):
            - task: "stack blocks"
            - stack_order: list of colors from bottom to top
            - objects_to_manipulate: list of "color block" strings
            - target_position: tuple (x, y) normalized to 0-1000 range (from camera click)
            - target_location: description string

        For sort task (2 steps):
            - task: "sort blocks"
            - block_positions: dict mapping block color -> (x, y) normalized position
            - objects_to_manipulate: list of "color block" strings
            - target_positions: dict mapping block color -> description string

        Returns None if user cancels.
    """
    root = tk.Tk()
    app = VLMInputGUI(root)
    root.mainloop()
    return app.get_result()
