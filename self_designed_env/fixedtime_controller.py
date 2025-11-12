class FixedTimeTrafficLight:
    """Fixed-time traffic light controller"""

    def __init__(self, phase_durations=[60, 60, 40, 40]):
        self.phase_durations = phase_durations
        self.num_phases = len(phase_durations)
        self.current_phase = 0
        self.phase_timer = 0

        self.phase_descriptions = [
            "North-South Straight Green",
            "East-West Straight Green",
            "North-South Left Turn Green",
            "East-West Left Turn Green"
        ]

    def step(self):
        """Update traffic light state"""
        self.phase_timer += 1
        if self.phase_timer >= self.phase_durations[self.current_phase]:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.phase_timer = 0
            print(f"🚦 Traffic light changed to: {self.phase_descriptions[self.current_phase]}")
        return self.current_phase

    def get_status(self):
        """Get current status"""
        return {
            'current_phase': self.current_phase,
            'phase_description': self.phase_descriptions[self.current_phase],
            'remaining_time': self.phase_durations[self.current_phase] - self.phase_timer
        }

    def reset(self):
        """Reset controller"""
        self.current_phase = 0
        self.phase_timer = 0