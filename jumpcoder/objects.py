from dataclasses import dataclass, field
from typing import List

@dataclass
class LineScore:
    token: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)

    def __len__(self): return len(self.token)

    def clean(self):
        token2 = []
        score2 = []
        for t, s in zip(self.token, self.score):
            if t.strip() != "" and t != '<s>':
                token2.append(t)
                score2.append(s)
        self.token = token2
        self.score = score2
        return self

    def dump(self):
        return [self.token, self.score]


@dataclass
class InfillingRecord:
    # current code lines after infilling
    cur_lines: List[str]
    # scores for current code lines
    generation_scores: List[LineScore]
    # infilling index
    index: int
    # delta = self.generation_scores[i] - PreviousCode.generation_scores[i]
    score_improvement: List[float]

