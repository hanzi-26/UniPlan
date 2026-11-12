"""Optional main-policy search-cost control, independent of dataset identity.

This is a compute-budget heuristic, not a quality certificate or a proof that
unseen candidates are infeasible. It never changes the probability threshold.
"""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class SearchBudget:
    max_extra_cost_fraction: float = .25

    @classmethod
    def from_config(cls, options):
        if not isinstance(options, dict):
            raise ValueError('main_search_budget must be an object')
        if set(options) - {'enabled', 'max_extra_cost_fraction'}:
            raise ValueError('unknown main_search_budget settings')
        if type(options.get('enabled', False)) is not bool:
            raise ValueError('main_search_budget.enabled must be boolean')
        fraction = options.get('max_extra_cost_fraction', .25)
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError('search cost fraction must be in (0,1]')
        return cls(float(fraction)) if options.get('enabled', False) else None

    def expansion(self, *, coarse_candidates, full_candidates, coarse_compute_ms,
                  incumbent_execution_ms, incumbent_upper_ms, remaining_ms):
        values = (coarse_compute_ms, incumbent_execution_ms, incumbent_upper_ms, remaining_ms)
        if (coarse_candidates <= 0 or full_candidates < coarse_candidates
                or any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError('invalid expansion cost inputs')
        # The existing recursive expansion evaluates the full bank again. Charge
        # that complete next pass; do not pretend coarse predictions are reused.
        estimated = coarse_compute_ms * full_candidates / coarse_candidates
        allowance = incumbent_execution_ms * self.max_extra_cost_fraction
        if estimated + incumbent_upper_ms > remaining_ms:
            reason = 'insufficient_remaining_budget'
        elif estimated > allowance:
            reason = 'estimated_search_cost_exceeds_allowance'
        else:
            reason = 'within_search_allowance'
        return dict(expand=reason == 'within_search_allowance', reason=reason,
            estimated_expansion_ms=estimated, extra_cost_allowance_ms=allowance,
            coarse_compute_ms=coarse_compute_ms, coarse_candidates=coarse_candidates,
            full_candidates=full_candidates, incumbent_execution_ms=incumbent_execution_ms,
            incumbent_upper_ms=incumbent_upper_ms, remaining_ms=remaining_ms,
            config=asdict(self), quality_threshold_relaxed=False,
            unseen_candidates_proven_infeasible=False)
