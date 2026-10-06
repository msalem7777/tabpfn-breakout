"""Install and record the analog target controller used by both policies."""
from breakout_pfn.analog_paddle_controller import install, smoke_test


class ControllerTrace:
    """Record actual paddle commands without choosing the policy target."""

    def __init__(self, env, mode='analog-target'):
        if mode != 'analog-target':
            raise ValueError('This demonstration uses the analog-target controller')
        install(self, env)
