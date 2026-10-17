"""Online MapAnything features for serving a geo_adapter="mapanything" pi0.5 (eval client -> this -> pi0.5).

The client (molmospaces PI_Policy, MV_PI_MAPANYTHING=1) sends the exterior render, its metric depth and
intrinsics as "observation/mapanything_rgb" / "_depth" / "_K". This computes the same features as the
training cache (MapAnythingFeatures, same layer), replaces those keys by "observation/geo_features_exterior"
and forwards the observation to the pi0.5 server. No extrinsics; a missing input is an error, never zeros.

  CUDA_VISIBLE_DEVICES=<gpu> /data/haotian/mapanything/venv/bin/python -m mapanything_features.proxy \
      --pi05-port 8094 --port 8093 [--layer final]
"""

import argparse
import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
import websockets.asyncio.server as ws_server
import websockets.frames

from mapanything_features.extractor import MapAnythingFeatures

KEYS = ("observation/mapanything_rgb", "observation/mapanything_depth", "observation/mapanything_K")


class MapAnythingProxy:
    def __init__(self, pi05_host: str, pi05_port: int, layer: str):
        self.features = MapAnythingFeatures(layer=layer)
        self.pi05 = websocket_client_policy.WebsocketClientPolicy(pi05_host, pi05_port)
        self.logged = False

    @property
    def metadata(self) -> dict:
        return self.pi05.get_server_metadata()

    def infer(self, obs: dict) -> dict:
        missing = [k for k in KEYS if k not in obs]
        if missing:
            raise KeyError(f"client must send {missing} (molmospaces PI_Policy: MV_PI_MAPANYTHING=1)")
        start = time.monotonic()
        rgb, depth, K = (np.asarray(obs.pop(k)) for k in KEYS)
        if not self.logged:
            logging.info("first call: rgb %s, depth %s (valid %.2f), K %s", rgb.shape, depth.shape, (depth > 0).mean(), K.round(2).tolist())
            self.logged = True
        obs["observation/geo_features_exterior"] = self.features(rgb, depth, K)[0]
        feature_ms = 1000 * (time.monotonic() - start)
        out = self.pi05.infer(obs)
        out["mapanything_ms"] = feature_ms
        return out


async def serve(policy: MapAnythingProxy, host: str, port: int):
    async def handler(websocket):
        logging.info("connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(policy.metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                start = time.monotonic()
                action = policy.infer(obs)
                action["server_timing"] = {"infer_ms": 1000 * (time.monotonic() - start)}
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="internal error")
                raise

    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async with ws_server.serve(handler, host, port, compression=None, max_size=None, process_request=health) as server:
        await server.serve_forever()


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi05-host", default="localhost")
    ap.add_argument("--pi05-port", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--layer", default="final", help="must match the training cache's layer")
    args = ap.parse_args()
    policy = MapAnythingProxy(args.pi05_host, args.pi05_port, args.layer)
    logging.info("serving pi0.5 + MapAnything features (%s) on port %d (pi0.5 at %s:%d)", args.layer, args.port, args.pi05_host, args.pi05_port)
    asyncio.run(serve(policy, "0.0.0.0", args.port))


if __name__ == "__main__":
    main()
