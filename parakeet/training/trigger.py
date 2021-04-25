class IntervalTrigger(object):
    def __init__(self, period: int , unit: str):
        if unit not in ("iteration", "epoch"):
            raise ValueError("unit should be 'iteration' or 'epoch'")
        self.period = period
        self.unit = unit
        
    def __call__(self, trainer):
        state = trainer.updater.state
        if self.unit == "epoch":
            fire = not (state.epoch % self.period)
        else:
            fire = not (state.iteration % self.iteration)
        return fire

  
def never_file_trigger(trainer):
    return False


def get_trigger(trigger):
    if trigger is None:
        return never_file_trigger
    if callable(trigger):
        return trigger
    else:
        trigger = IntervalTrigger(*trigger)
        return trigger