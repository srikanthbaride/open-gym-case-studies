from dataclasses import dataclass

@dataclass
class Sched:
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_episodes: int = 3000

    def epsilon(self, t: int) -> float:
        if t >= self.eps_decay_episodes:
            return self.eps_end
        return self.eps_start - (self.eps_start - self.eps_end) * (t / self.eps_decay_episodes)
