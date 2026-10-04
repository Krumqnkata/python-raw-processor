"""Launch the RAW Studio desktop application: python raw_processor.py."""
import sys


def main() -> None:
    try:
        import cv2
        import customtkinter as ctk
        from app_gui import AppGUI
    except ImportError as error:
        print(f"Липсва зависимост: {error}\nИзпълни: python -m pip install -r requirements.txt",
              file=sys.stderr)
        raise SystemExit(1) from error
    # LibRaw/OpenCV may use their own native threads; batches are intentionally serial.
    cv2.setNumThreads(2)
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = AppGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
