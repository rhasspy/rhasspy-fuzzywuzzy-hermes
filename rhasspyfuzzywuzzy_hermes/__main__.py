"""Hermes MQTT service for rhasspy fuzzywuzzy"""
import argparse
import logging
import os
import threading
import time
import typing
from pathlib import Path
from uuid import uuid4

import paho.mqtt.client as mqtt
from rhasspyhermes.nlu import NluTrain

from . import NluHermesMqtt

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspy-fuzzywuzzy-hermes")
    parser.add_argument("--examples", help="Path to examples JSON file")
    parser.add_argument("--intent-graph", help="Path to intent graph JSON file")
    parser.add_argument(
        "--sentences",
        action="append",
        help="Watch sentences.ini file(s) for changes and re-train",
    )
    parser.add_argument(
        "--slots", action="append", help="Directories with static slot text files"
    )
    parser.add_argument(
        "--slot-programs", action="append", help="Directories with slot programs"
    )
    parser.add_argument(
        "--watch-delay",
        type=float,
        default=1.0,
        help="Seconds between polling sentence file(s) for training",
    )
    parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )
    parser.add_argument(
        "--language", default="en", help="Language used for number replacement"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    try:
        # Convert to Paths
        if args.sentences:
            args.sentences = [Path(p) for p in args.sentences]

        if args.examples:
            args.examples = Path(args.examples)

        if args.intent_graph:
            args.intent_graph = Path(args.intent_graph)

        if args.slots:
            args.slots = [Path(p) for p in args.slots]

        if args.slot_programs:
            args.slot_programs = [Path(p) for p in args.slot_programs]

        # Listen for messages
        client = mqtt.Client()
        hermes = NluHermesMqtt(
            client,
            intent_graph_path=args.intent_graph,
            examples_path=args.examples,
            sentences=args.sentences,
            slots_dirs=args.slots,
            slot_programs_dirs=args.slot_programs,
            language=args.language,
            siteIds=args.siteId,
        )

        if args.sentences and (args.watch_delay > 0):
            # Start polling thread
            threading.Thread(
                target=poll_sentences,
                args=(args.sentences, args.watch_delay, args.examples, hermes),
                daemon=True,
            ).start()

        def on_disconnect(client, userdata, flags, rc):
            try:
                # Automatically reconnect
                _LOGGER.info("Disconnected. Trying to reconnect...")
                client.reconnect()
            except Exception:
                logging.exception("on_disconnect")

        # Connect
        client.on_connect = hermes.on_connect
        client.on_disconnect = on_disconnect
        client.on_message = hermes.on_message

        _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
        client.connect(args.host, args.port)

        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")


# -----------------------------------------------------------------------------


def poll_sentences(
    sentences_paths: typing.List[Path],
    delay_seconds: float,
    graph_path: Path,
    hermes: NluHermesMqtt,
):
    """Watch sentences for changes and retrain."""
    last_timestamps: typing.Dict[Path, int] = {}

    while True:
        time.sleep(delay_seconds)
        try:
            retrain = False
            for sentences_path in sentences_paths:
                timestamp = os.stat(sentences_path).st_mtime_ns
                last_timestamp = last_timestamps.get(sentences_path)
                if (last_timestamp is not None) and (timestamp != last_timestamp):
                    retrain = True

                last_timestamps[sentences_path] = timestamp

            if retrain:
                _LOGGER.debug("Re-training")
                sentences_dict: typing.Dict[str, str] = {}
                for sentences_path in sentences_paths:
                    sentences_key = str(sentences_path)
                    sentences_dict[sentences_key] = sentences_path.read_text()

                result = hermes.handle_train(
                    NluTrain(id=str(uuid4()), sentences=sentences_dict)
                )
                hermes.publish(result)
        except Exception:
            _LOGGER.exception("poll_sentences")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
