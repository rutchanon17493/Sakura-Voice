"""
Sakura Voice Agent — entry point.

Usage:
  python run.py                         Run the local agent (default)
  python run.py --agent local           Run the local agent
  python run.py --agent sarvam          Run the Sarvam AI agent
  python run.py --agent indic           Run the fully-local Indic agent
  python run.py --list-languages        List all Sarvam-supported languages
  python run.py --list-indic-languages  List all Indic agent supported languages
  python run.py --list-devices          List all audio input devices
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Sakura Voice Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["local", "sarvam", "indic"],
        default="local",
        help="Which agent to run (default: local)",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all Sarvam AI supported languages and exit",
    )
    parser.add_argument(
        "--list-indic-languages",
        action="store_true",
        help="List all Indic agent supported languages and exit",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_languages:
        from sakura.languages import print_languages
        print_languages()
        sys.exit(0)

    if args.list_indic_languages:
        from sakura.languages.indic import print_indic_languages
        print_indic_languages()
        sys.exit(0)

    if args.list_devices:
        from sakura.audio import AudioEngine
        AudioEngine.list_devices()
        sys.exit(0)

    if args.agent == "sarvam":
        from agents.sarvam_agent import SarvamVoiceAgent
        SarvamVoiceAgent().run()
    elif args.agent == "indic":
        from agents.indic_agent import IndicVoiceAgent
        IndicVoiceAgent().run()
    else:
        from agents.local_agent import LocalVoiceAgent
        LocalVoiceAgent().run()


if __name__ == "__main__":
    main()
