# -*- coding: utf-8 -*-
"""Base class for testing the moe examples."""
from builtins import object
import pytest


class MoeExampleTestCase(object):

    """Base class for testing the moe examples."""

    @classmethod
    @pytest.fixture(autouse=True, scope='class')
    def create_webapp(cls):
        """Create a mocked webapp and store it in cls.testapp."""
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        cls.testapp = TestApp(app)
