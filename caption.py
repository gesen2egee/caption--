# -*- coding: utf-8 -*-
import argparse
import sys
import os
import warnings
import webbrowser

# silence some noisy third-party warnings
os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_fwd")
warnings.filterwarnings("ignore", message="Failed to import flet")
warnings.filterwarnings("ignore", message="Token indices sequence length")

# [GPU Fix] 嘗試載入 pip 安裝的 NVIDIA dll
if os.name == 'nt':
    try:
        import nvidia.cudnn
        import nvidia.cublas
        libs = [
            os.path.dirname(nvidia.cudnn.__file__),
            os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin"),
            os.path.dirname(nvidia.cublas.__file__),
            os.path.join(os.path.dirname(nvidia.cublas.__file__), "bin"),
        ]
        for lib in libs:
            if os.path.exists(lib):
                os.add_dll_directory(lib)
    except Exception:
        pass

def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Caption app launcher")
    parser.add_argument(
        "--ui",
        choices=("qt", "web", "shell", "service"),
        default=str(os.getenv("CAPTION_UI_MODE", "qt")).strip().lower() or "qt",
        help="Choose the UI host mode.",
    )
    parser.add_argument(
        "--runtime-host",
        default=str(os.getenv("CAPTION_RUNTIME_HTTP_HOST", "127.0.0.1")).strip() or "127.0.0.1",
        help="Runtime bridge host for web or shell modes.",
    )
    parser.add_argument(
        "--runtime-port",
        type=int,
        default=int(str(os.getenv("CAPTION_RUNTIME_HTTP_PORT", "8765")).strip() or "8765"),
        help="Runtime bridge port for web or shell modes.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the runtime URL in the default browser after startup.",
    )
    parser.add_argument(
        "--auto-reload-services",
        action="store_true",
        help="Enable automatic reload for extracted backend service modules.",
    )
    return parser.parse_known_args(argv)


def configure_runtime_env(args: argparse.Namespace) -> None:
    if args.ui not in {"web", "shell", "service"}:
        return
    os.environ["CAPTION_RUNTIME_HTTP"] = "1"
    os.environ["CAPTION_RUNTIME_HTTP_HOST"] = str(args.runtime_host or "127.0.0.1")
    os.environ["CAPTION_RUNTIME_HTTP_PORT"] = str(int(args.runtime_port or 8765))
    if args.auto_reload_services:
        os.environ["CAPTION_RUNTIME_AUTO_RELOAD_SERVICES"] = "1"


def runtime_url(args: argparse.Namespace) -> str:
    return f"http://{args.runtime_host}:{int(args.runtime_port)}"  # pragma: no cover - trivial helper


def should_show_legacy_window(args: argparse.Namespace) -> bool:
    return args.ui in {"qt", "shell"}


def run_qt_host(args: argparse.Namespace, qt_args: list[str]) -> int:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from lib.ui.main_window import MainWindow

    app = QApplication([sys.argv[0], *qt_args])
    if not should_show_legacy_window(args):
        app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    if should_show_legacy_window(args):
        window.show()
    else:
        window.hide()

    if args.open_browser and args.ui == "web":
        QTimer.singleShot(900, lambda: webbrowser.open(runtime_url(args)))
    return app.exec()


def run_headless_host(args: argparse.Namespace) -> int:
    from lib.runtime.backend_host import RuntimeBackendHost

    host = RuntimeBackendHost()
    if args.open_browser and args.ui == "web":
        webbrowser.open(runtime_url(args))
    host.wait_forever()
    return 0


if __name__ == "__main__":
    args, qt_args = parse_args(sys.argv[1:])
    configure_runtime_env(args)
    if args.ui in {"web", "service"}:
        sys.exit(run_headless_host(args))
    sys.exit(run_qt_host(args, qt_args))
