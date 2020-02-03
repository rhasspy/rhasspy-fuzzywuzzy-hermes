"""Hermes MQTT service for rhasspy fuzzywuzzy"""
import argparse
import json
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
        if args.examples:
            args.examples = Path(args.examples)

        if args.intent_graph:
            args.intent_graph = Path(args.intent_graph)

        # Listen for messages
        client = mqtt.Client()
        hermes = NluHermesMqtt(
            client,
            intent_graph_path=args.intent_graph,
            examples_path=args.examples,
            language=args.language,
            siteIds=args.siteId,
        )

        if args.intent_graph and (args.watch_delay > 0):
            # Start polling thread
            threading.Thread(
                target=poll_intent_graph,
                args=(args.intent_graph, args.watch_delay, hermes),
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


def poll_intent_graph(graph_path: Path, delay_seconds: float, hermes: NluHermesMqtt):
    """Watch sentences for changes and retrain."""
    last_timestamps: typing.Dict[Path, int] = {}

    while True:
        time.sleep(delay_seconds)
        try:
            retrain = False
            for path in [graph_path]:
                timestamp = os.stat(path).st_mtime_ns
                last_timestamp = last_timestamps.get(path)
                if (last_timestamp is not None) and (timestamp != last_timestamp):
                    retrain = True

                last_timestamps[path] = timestamp

            if retrain:
                _LOGGER.debug("Re-training")
                with open(graph_path, "r") as graph_file:
                    graph_dict = json.load(graph_file)
                    result = hermes.handle_train(
                        NluTrain(id=str(uuid4()), graph_dict=graph_dict)
                    )
                    hermes.publish(result)
        except Exception:
            _LOGGER.exception("poll_sentences")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
