class DispatchRotator:
    """
    Rotating dispatch phase generator.

    Cycles through 4 phases where each phase starts at a different tier,
    guaranteeing no tier waits more than 3 dispatch cycles before being
    serviced first. This is the exact algorithm from scx_flow.
    """

    # The 4 dispatch phases, each starting at a different tier
    # Values are tier indices (0=Priority, 1=Normal, 2=Low, 3=Deficit)
    PHASES = [
        [0, 1, 2, 3],  # Priority -> Normal -> Low -> Deficit
        [1, 2, 3, 0],  # Normal -> Low -> Deficit -> Priority
        [2, 3, 0, 1],  # Low -> Deficit -> Priority -> Normal
        [3, 0, 1, 2],  # Deficit -> Priority -> Normal -> Low
    ]

    def __init__(self, n_tiers=4):
        self.n_tiers = n_tiers
        self._phase = 0

    @property
    def phase(self):
        """Return the current phase index (0-3)."""
        return self._phase

    def current_order(self):
        """Return the tier dispatch order for the current phase."""
        return self.PHASES[self._phase % 4]

    def advance(self):
        """Advance to the next dispatch phase."""
        self._phase = (self._phase + 1) % 4
        return self.current_order()

    def reset(self):
        """Reset the rotator back to phase 0."""
        self._phase = 0
