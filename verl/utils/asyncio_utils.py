import asyncio


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the current event loop, creating one if none exists.

    Under uvloop's event loop policy (installed transitively by sglang) on
    Python 3.10+, ``asyncio.get_event_loop()`` raises ``RuntimeError`` when no
    loop is set for the current thread instead of creating one. This helper
    restores the historical get-or-create behaviour that the synchronous
    ``run_until_complete`` call sites rely on.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
