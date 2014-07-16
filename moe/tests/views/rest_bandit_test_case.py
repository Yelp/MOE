# -*- coding: utf-8 -*-
"""Base class for testing the REST interface against the interface."""
import testify as T

from moe.tests.bandit.bandit_test_case import BanditTestCase


class RestBanditTestCase(BanditTestCase):

    """Base class for testing the REST interface against the interface."""

    endpoint = None

    @T.class_setup
    def create_webapp(self):
        """Create a mocked webapp and store it in self.testapp."""
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        self.testapp = TestApp(app)
