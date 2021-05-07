import contextlib

OBSERVATIONS = None

@contextlib.contextmanager
def scope(observations):
    # make `observation` the target to report to.
    # it is basically a dictionary that stores temporary observations
    global OBSERVATIONS
    old = OBSERVATIONS
    OBSERVATIONS = observations

    try:
        yield
    finally:
        OBSERVATIONS = old    

def get_observations():
    global OBSERVATIONS
    return OBSERVATIONS

def report(name, value):
    # a simple function to report named value
    # you can use it everywhere, it will get the default target and writ to it
    # you can think of it as std.out
    observations = get_observations()
    if observations is None:
        return
    else:
        observations[name] = value
