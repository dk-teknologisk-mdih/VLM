"""Draggable block canvas widget with visual effects."""

import tkinter as tk

from .constants import BLOCK_COLORS_MAP, DISPLAY_LABELS


class DraggableBlock:
    """A draggable block on a canvas with visual effects."""

    COLORS = BLOCK_COLORS_MAP
    DISPLAY_LABELS = DISPLAY_LABELS

    def __init__(self, canvas, color, x, y, width=100, height=50):
        self.canvas = canvas
        self.color = color
        self.width = width
        self.height = height
        self.colors = self.COLORS[color]
        self.is_dragging = False
        self.label = self.DISPLAY_LABELS.get(color, color.upper())

        # Text color: use light text for dark blocks, dark text for yellow
        self.text_color = "#1a1a2e" if color == "yellow" else "white"

        # Create shadow
        self.shadow = canvas.create_rectangle(
            x + 4, y + 4, x + width + 4, y + height + 4,
            fill=self.colors["shadow"], outline=""
        )

        # Create main rectangle
        self.rect = canvas.create_rectangle(
            x, y, x + width, y + height,
            fill=self.colors["fill"], outline="white", width=3
        )

        # Create shine effect (top highlight)
        self.shine = canvas.create_rectangle(
            x + 3, y + 3, x + width - 3, y + 12,
            fill="", outline=""
        )

        # Create text with shadow
        self.text_shadow = canvas.create_text(
            x + width // 2 + 1, y + height // 2 + 1,
            text=self.label, fill="#333333", font=("Arial", 10, "bold")
        )
        self.text = canvas.create_text(
            x + width // 2, y + height // 2,
            text=self.label, fill=self.text_color, font=("Arial", 10, "bold")
        )

        # Bind events for all elements
        for item in [self.rect, self.text, self.text_shadow, self.shine]:
            canvas.tag_bind(item, "<Button-1>", self._on_press)
            canvas.tag_bind(item, "<B1-Motion>", self._on_drag)
            canvas.tag_bind(item, "<ButtonRelease-1>", self._on_release)
            canvas.tag_bind(item, "<Enter>", self._on_enter)
            canvas.tag_bind(item, "<Leave>", self._on_leave)

        self.drag_data = {"x": 0, "y": 0}
        self.on_drop_callback = None
        self.on_drag_callback = None

    def _on_enter(self, event):
        """Hover effect."""
        if not self.is_dragging:
            self.canvas.itemconfig(self.rect, fill=self.colors["hover"])
            self.canvas.config(cursor="hand2")

    def _on_leave(self, event):
        """Remove hover effect."""
        if not self.is_dragging:
            self.canvas.itemconfig(self.rect, fill=self.colors["fill"])
            self.canvas.config(cursor="")

    def _on_press(self, event):
        """Store initial position for drag."""
        self.is_dragging = True
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        # Raise all elements
        for item in [self.shadow, self.rect, self.shine, self.text_shadow, self.text]:
            self.canvas.tag_raise(item)
        # Enlarge effect
        self.canvas.itemconfig(self.rect, width=4, outline=self.colors["glow"])
        self.canvas.config(cursor="fleur")

    def _on_drag(self, event):
        """Handle dragging."""
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        for item in [self.shadow, self.rect, self.shine, self.text_shadow, self.text]:
            self.canvas.move(item, dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        if self.on_drag_callback:
            self.on_drag_callback(self, event.x, event.y)

    def _on_release(self, event):
        """Handle drop."""
        self.is_dragging = False
        self.canvas.itemconfig(self.rect, width=3, outline="white", fill=self.colors["fill"])
        self.canvas.config(cursor="")
        if self.on_drop_callback:
            self.on_drop_callback(self, event.x, event.y)

    def get_center(self):
        """Get center coordinates of the block."""
        coords = self.canvas.coords(self.rect)
        return (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2

    def move_to(self, x, y, animate=False):
        """Move block center to specified position."""
        cx, cy = self.get_center()
        dx = x - cx
        dy = y - cy
        for item in [self.shadow, self.rect, self.shine, self.text_shadow, self.text]:
            self.canvas.move(item, dx, dy)

    def get_y(self):
        """Get top y coordinate."""
        return self.canvas.coords(self.rect)[1]

    def highlight(self, on=True):
        """Highlight block as potential swap target."""
        if on:
            self.canvas.itemconfig(self.rect, outline=self.colors["glow"], width=4)
        else:
            self.canvas.itemconfig(self.rect, outline="white", width=3)
