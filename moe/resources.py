# -*- coding: utf-8 -*-
"""Resources class."""
from builtins import object


class Root(object):

    """Resources Root. This is the base view class."""

    def __init__(self, request):
        """Initialization for the root resource."""
        self.request = request
