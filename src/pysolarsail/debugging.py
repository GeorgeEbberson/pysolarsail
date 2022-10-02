"""
Logger config and debug telemetry stuff.
"""
import pandas as pd

# The telemetry dataframe and a "buffer", which is a dictionary of values
# to add to the telemetry next time inc_time is called.
_TELEM = None
_BUFFER = None


def inc_time(np_time):
    """Flush the logging buffer and start a new timestep."""
    # Flush the buffer.
    global _TELEM
    global _BUFFER

    # Only flush the buffer if we have more than just "time".
    if len(_BUFFER) > 1:
        _TELEM = pd.concat((_TELEM, pd.DataFrame(_BUFFER)))

    # Now reset the buffer and start again.
    _BUFFER = {"time": [np_time]}


def telemetry(**kwargs):
    """Adds value to the telemetry buffer in name."""
    global _TELEM
    global _BUFFER

    for name, value in kwargs.items():
        if name in _BUFFER.keys():
            raise ValueError(f"Already logged {name} this timestep")
        _BUFFER[name] = value


def telemetry_3vec(**kwargs):
    """telemetry a 3-vector append x, y, z to their names."""
    for name, arr in kwargs.items():
        for idx, suffix in enumerate(("x", "y", "z")):
            telemetry(**{f"{name}_{suffix}": arr[idx]})


def init_telemetry(dtime):
    """Initialises telemetry, should only be called once."""
    global _TELEM
    global _BUFFER
    if _TELEM is not None:
        raise RuntimeError("Cannot reinitialise telemetry!")
    _TELEM = pd.DataFrame()
    _BUFFER = {"time": [dtime]}


def dump_telemetry(filename):
    """Write the telemetry to a file."""
    global _TELEM
    _TELEM.to_csv(filename, index=False)


def get_telemetry():
    """Return telemetry."""
    global _TELEM
    return _TELEM


def reset_telemetry():
    """Resets the telemetry state so init can be called again."""
    global _TELEM
    global _BUFFER
    _TELEM = None
    _BUFFER = None


def get_dump_and_reset_telemetry(filename):
    """To be called when you want to get the values and start afresh."""
    dump_telemetry(filename)
    telem = get_telemetry()
    reset_telemetry()
    return telem
