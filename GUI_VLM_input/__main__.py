"""Allow running the GUI as: python -m GUI_VLM_input"""

from .vlm_input_gui import get_user_input

if __name__ == "__main__":
    result = get_user_input()
    if result:
        print("Selected configuration:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("User cancelled")
