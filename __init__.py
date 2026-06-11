# core/__init__.py


import asyncio
import logging

from app.services.refresh_service import (
    should_refresh,
    run_refresh_pipeline,
)

logger = logging.getLogger(__name__)


async def refresh_loop():

    while True:

        try:

            if should_refresh():

                logger.info(
                    "Knowledge base older than 3 days. Refreshing..."
                )

                await asyncio.to_thread(
                    run_refresh_pipeline
                )

            else:
                logger.info(
                    "Knowledge base still fresh."
                )

        except Exception as e:
            logger.exception(
                f"Refresh failed: {e}"
            )

        await asyncio.sleep(3600)