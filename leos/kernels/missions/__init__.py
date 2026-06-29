"""
leos/kernels/missions/

One module per spacecraft mission. Each module exposes:

    get_kernel_urls(time=None, time_range=None) -> dict[filename, url]

with that exact signature, even if a given mission ignores time/time_range
internally (most do -- only MAVEN currently uses them, to pick weekly
attitude CK files). Keeping the signature uniform is what lets
fetch_kernels.py dispatch through one MISSION_RESOLVERS table instead of
special-casing each mission.

To add a new mission:
  1. Create fetch_<name>_kernels.py here (or under a planet subpackage like
     Mars/) with a get_kernel_urls() function.
  2. Import it in fetch_kernels.py and add one line to MISSION_RESOLVERS.
"""
