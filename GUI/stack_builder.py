"""Visual stack builder widget: drag blocks from pool into stack slots."""

import tkinter as tk

from .constants import PALETTE, ALL_BLOCK_COLORS
from .draggable_block import DraggableBlock


class StackBuilder:
    """Visual stack builder: drag blocks from pool into 4 stack slots."""

    MAX_STACK = 4

    def __init__(self, parent, width=340, height=480):
        self.frame = tk.Frame(parent, bg=PALETTE["bg_dark"])
        self.width = width
        self.height = height

        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg=PALETTE["bg_dark"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Draw decorative grid background
        for i in range(0, height, 25):
            self.canvas.create_line(0, i, width, i, fill=PALETTE["bg_mid"], width=1)
        for i in range(0, width, 25):
            self.canvas.create_line(i, 0, i, height, fill=PALETTE["bg_mid"], width=1)

        # Title
        self.canvas.create_text(
            width // 2 + 1, 16, text="🎮 DRAG BLOCKS INTO STACK",
            fill=PALETTE["bg_dark"], font=("Arial", 11, "bold")
        )
        self.canvas.create_text(
            width // 2, 15, text="🎮 DRAG BLOCKS INTO STACK",
            fill=PALETTE["accent_cyan"], font=("Arial", 11, "bold")
        )
        self.canvas.create_text(
            width // 2, 33, text="✨ Up to 4 blocks (bottom → top) ✨",
            fill=PALETTE["accent_pink"], font=("Arial", 9)
        )

        # --- Stack area (middle, centered) ---
        stack_x = width // 2
        stack_base_y = 270  # Base of the stack in the middle zone

        # Platform
        self.canvas.create_rectangle(stack_x - 60, stack_base_y - 3, stack_x + 60, stack_base_y + 15,
                                     fill="", outline=PALETTE["accent_purple"], width=2)
        self.canvas.create_rectangle(stack_x - 55, stack_base_y, stack_x + 55, stack_base_y + 12,
                                     fill=PALETTE["accent_purple"], outline=PALETTE["accent_cyan"], width=2)

        # 4 stack slots
        self.slot_height = 50
        self.slot_x = stack_x
        self.slots = []
        self.slot_indicators = []

        for i in range(self.MAX_STACK):
            slot_y = stack_base_y - 28 - (i * self.slot_height)
            self.slots.append(slot_y)
            indicator = self.canvas.create_rectangle(
                stack_x - 50, slot_y - 20, stack_x + 50, slot_y + 20,
                outline=PALETTE["accent_pink"], dash=(6, 3), width=2
            )
            self.slot_indicators.append(indicator)
            # Slot number
            self.canvas.create_text(
                stack_x - 62, slot_y + 1, text=str(i + 1),
                fill=PALETTE["bg_dark"], font=("Arial", 12, "bold")
            )
            self.canvas.create_text(
                stack_x - 63, slot_y, text=str(i + 1),
                fill=PALETTE["accent_cyan"], font=("Arial", 12, "bold")
            )

        self.canvas.create_text(stack_x + 63, self.slots[-1], text="⬆ TOP",
                               fill=PALETTE["accent_cyan"], font=("Arial", 8, "bold"))
        self.canvas.create_text(stack_x + 63, self.slots[0], text="⬇ BTM",
                               fill=PALETTE["accent_orange"], font=("Arial", 8, "bold"))

        # Separator line
        sep_y = 295
        self.canvas.create_line(20, sep_y, width - 20, sep_y,
                               fill=PALETTE["accent_purple"], width=1, dash=(4, 4))

        # --- Pool area (bottom, two rows of 3) ---
        self.canvas.create_text(width // 2, 312, text="AVAILABLE BLOCKS",
                               fill=PALETTE["accent_yellow"], font=("Arial", 9, "bold"))

        # Two rows of 3 blocks, centered
        block_w, block_h = 95, 42
        col_spacing = 108
        row_spacing = 55
        pool_start_y = 345
        pool_start_x = width // 2 - col_spacing  # leftmost column center

        # Create all blocks in the pool
        self.blocks = {}
        self.block_in_slot = {}  # color -> slot index (None if in pool)
        self.slot_contents = {}  # slot index -> color (which block is in each slot)
        self.pool_positions = {}  # color -> (x, y) home position in pool

        for i, color in enumerate(ALL_BLOCK_COLORS):
            row = i // 3
            col = i % 3
            px = pool_start_x + col * col_spacing
            py = pool_start_y + row * row_spacing
            self.pool_positions[color] = (px, py)
            self.block_in_slot[color] = None

            block = DraggableBlock(self.canvas, color, px - block_w // 2, py - block_h // 2,
                                   width=block_w, height=block_h)
            block.on_drop_callback = self._on_block_drop
            block.on_drag_callback = self._on_block_drag
            self.blocks[color] = block

    def _on_block_drag(self, block, x, y):
        """Highlight the slot being hovered over."""
        hover_slot = None
        for i, slot_y in enumerate(self.slots):
            if abs(x - self.slot_x) < 70 and abs(y - slot_y) < 30:
                hover_slot = i
                break

        for color, b in self.blocks.items():
            if b is not block:
                slot = self.block_in_slot[color]
                if hover_slot is not None and slot == hover_slot:
                    b.highlight(True)
                else:
                    b.highlight(False)

    def _on_block_drop(self, dropped_block, x, y):
        """Handle block drop: place into slot, swap, or return to pool."""
        for b in self.blocks.values():
            b.highlight(False)

        # Find which block was dropped
        dropped_color = None
        for color, block in self.blocks.items():
            if block is dropped_block:
                dropped_color = color
                break

        original_slot = self.block_in_slot[dropped_color]

        # Check if dropped on a stack slot
        target_slot = None
        for i, slot_y in enumerate(self.slots):
            if abs(x - self.slot_x) < 70 and abs(y - slot_y) < 35:
                target_slot = i
                break

        if target_slot is not None:
            # Check if another block is in that slot
            occupant_color = self.slot_contents.get(target_slot)

            if occupant_color and occupant_color != dropped_color:
                # Swap: move occupant to dropped block's original position
                if original_slot is not None:
                    # Dropped block was in a slot -> swap slots
                    self.blocks[occupant_color].move_to(self.slot_x, self.slots[original_slot])
                    self.block_in_slot[occupant_color] = original_slot
                    self.slot_contents[original_slot] = occupant_color
                else:
                    # Dropped block was in pool -> send occupant back to pool
                    home = self.pool_positions[occupant_color]
                    self.blocks[occupant_color].move_to(home[0], home[1])
                    self.block_in_slot[occupant_color] = None
                    if original_slot in self.slot_contents:
                        del self.slot_contents[original_slot]
            elif original_slot is not None and original_slot != target_slot:
                # Moving between slots, clear old slot
                if self.slot_contents.get(original_slot) == dropped_color:
                    del self.slot_contents[original_slot]

            # Place dropped block in target slot
            dropped_block.move_to(self.slot_x, self.slots[target_slot])
            self.block_in_slot[dropped_color] = target_slot
            self.slot_contents[target_slot] = dropped_color
        else:
            # Dropped outside stack area -> return to pool
            if original_slot is not None:
                if self.slot_contents.get(original_slot) == dropped_color:
                    del self.slot_contents[original_slot]
            self.block_in_slot[dropped_color] = None
            home = self.pool_positions[dropped_color]
            dropped_block.move_to(home[0], home[1])

    def get_stack_order(self):
        """Get the stack order from bottom to top (only filled slots)."""
        order = []
        for i in range(self.MAX_STACK):
            color = self.slot_contents.get(i)
            if color:
                order.append(color)
        return order

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
