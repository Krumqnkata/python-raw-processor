"""Launch the RAW Studio desktop application: python raw_processor.py."""
import sys


def main() -> None:
    if sys.argv[1:2] == ["--smoke-test"]:
        try:
            from packaging_smoke import smoke
            smoke(*sys.argv[2:4])
        except Exception:
            import traceback
            from pathlib import Path
            directory=Path(sys.argv[3]);directory.mkdir(parents=True,exist_ok=True)
            (directory/'smoke-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            raise SystemExit(1)
        return
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
