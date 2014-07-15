# -*- coding: utf-8 -*-
"""Base class for testing the moe examples."""
import testify as T


class MoeExampleTestCase(T.TestCase):

    """Base class for testing the moe examples."""

    @T.class_setup
    def create_webapp(self):
        """Create a mocked webapp and store it in self.testapp."""
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        self.testapp = TestApp(app)
