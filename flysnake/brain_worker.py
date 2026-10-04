"""Process boundary: the neural process never receives a SnakeGame or a board."""
from __future__ import annotations

import traceback


def brain_entry(connection):
    try:
        from .neural import RealFlyBrain

        def progress(message):
            connection.send({"type": "progress", "message": message})

        brain = RealFlyBrain(progress=progress, learning=True)
        connection.send({"type": "ready", "result": brain.metadata()})
        while True:
            request = connection.recv()
            operation = request["operation"]
            if operation == "close":
                break
            if operation == "observe":
                result = brain.observe(request["rgb"])
            elif operation == "feedback":
                result = brain.feedback(request["kind"], request["rgb"])
            elif operation == "checkpoint":
                result = brain.checkpoint(request["path"])
            elif operation == "restore":
                brain.restore(request["path"])
                result = brain.metadata()
            elif operation == "reset_episode":
                reset = brain.reset_episode()
                result = {**brain.metadata(), "last_reset": reset}
            elif operation == "metadata":
                result = brain.metadata()
            else:
                raise ValueError(f"Unsupported neural operation: {operation}")
            connection.send({"type": "result", "result": result})
    except EOFError:
        pass
    except Exception as error:
        try:
            connection.send({"type": "error", "message": str(error), "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
