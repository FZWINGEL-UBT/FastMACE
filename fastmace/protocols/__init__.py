"""Temperature schedules and run protocols such as anneal (SPEC §7.2, §7.6 "Schedules", §10.5).

``ramp``, ``hold`` and ``+`` build a ``Schedule``; ``ScheduleState`` is its position (restart data);
``PROFILES`` holds today's profiles as data. Imports neither torch nor MACE.
"""

from fastmace.protocols.profiles import PROFILES, Profile, load_profile, profile_from_data
from fastmace.protocols.schedule import (
    ResolvedSchedule,
    Schedule,
    ScheduleState,
    Segment,
    TargetFill,
    hold,
    output_steps,
    ramp,
    segment_from_data,
)

__all__ = ["PROFILES", "Profile", "ResolvedSchedule", "Schedule", "ScheduleState", "Segment", "TargetFill", "hold",
           "load_profile", "output_steps", "profile_from_data", "ramp", "segment_from_data"]
