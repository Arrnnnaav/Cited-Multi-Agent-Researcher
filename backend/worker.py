"""Eval job worker.  Run: python -m backend.worker"""

import asyncio
import logging
import os

from backend import jobs, redis_layer
from backend.eval import runner as eval_runner

log = logging.getLogger("backend.worker")


async def handle(kind: str) -> dict:
    results = await eval_runner.run()
    return {
        "num_queries": len(results),
        "report": str(eval_runner.latest_report_path()),
    }


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    r = await redis_layer.get_redis()
    if r is None:
        raise SystemExit("REDIS_URL not reachable; the eval worker needs Redis")
    consumer = f"worker-{os.getpid()}"
    log.info("eval worker %s waiting for jobs", consumer)
    while True:
        await jobs.process_once(r, consumer, handle, block_ms=5000)


if __name__ == "__main__":
    asyncio.run(main())
