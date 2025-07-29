# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe
from ..run import RunProfile
from .data import RunProfileData

class RunGenerator:
    def __init__(self, worker, span, profile_data: RunProfileData):
        self.worker = worker
        self.span = span
        self.profile_data = profile_data

    def generate_run_profile(self):
        profile_run = RunProfile(self.worker, self.span)
        profile_run.tid2tree = self.profile_data.tid2tree
        profile_run.pl_tid2tree = self.profile_data.pl_tid2tree
        return profile_run
