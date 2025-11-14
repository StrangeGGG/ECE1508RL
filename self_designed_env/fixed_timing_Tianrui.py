class FixedCycleTrafficLight:
    """
    Improved fixed-time traffic light controller.
    - Supports arbitrary number of phases (straight, left, all-red, etc.)
    - Compatible with variable simulation time steps (dt)
    - Provides detailed phase information and querying of green directions
    """

    def __init__(self):
            
        phases = [
                {'name': 'NS_Straight', 'duration': 60, 'green_directions': ['N', 'S']},
                {'name': 'EW_Straight', 'duration': 60, 'green_directions': ['E', 'W']},
                {'name': 'NS_Left', 'duration': 40, 'green_directions': ['N_left', 'S_left']},
                {'name': 'EW_Left', 'duration': 40, 'green_directions': ['E_left', 'W_left']},
                ]

        self.phases = phases
        self.num_phases = len(phases)
        self.current_phase = 0
        self.phase_timer = 0.0

    def step(self, dt=1.0):
        """
        Advance the controller by dt seconds.
        Returns the current phase after potential change.
        """
        self.phase_timer += dt
        current_duration = self.phases[self.current_phase]['duration']

        if self.phase_timer >= current_duration:
            # move to next phase
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.phase_timer = 0.0
            print(f"🚦 Switched to phase {self.current_phase}: {self.phases[self.current_phase]['name']}")

        return self.current_phase
    
    def get_status(self):
        """Return current phase information as a dictionary."""
        phase = self.phases[self.current_phase]
        remaining = round(phase["duration"] - self.phase_timer, 1)
        # Backward-compatible keys for your visualizer
        return {
            "phase_index": self.current_phase,
            "phase_name": phase["name"],
            "remaining_time": remaining,
            "green_directions": phase["green_directions"],
            "current_phase": self.current_phase,
            "phase_description": phase["name"],
        }


    def is_green(self, direction):
        """Check whether a given direction (e.g., 'N', 'S_left') currently has green."""
        return direction in self.phases[self.current_phase]['green_directions']

    def reset(self):
        """Reset to the first phase."""
        self.current_phase = 0
        self.phase_timer = 0.0
