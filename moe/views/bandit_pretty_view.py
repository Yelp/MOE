# -*- coding: utf-8 -*-
"""A class to encapsulate Bandit 'pretty' views."""

from moe.bandit.constant import DEFAULT_BANDIT_HISTORICAL_INFO
from moe.views.pretty_view import PrettyView


class BanditPrettyView(PrettyView):

    """A class to encapsulate Bandit 'pretty' views.

    See :class:`moe.views.pretty_view.PrettyView` superclass for more details.

    """

    _pretty_default_historical_info = DEFAULT_BANDIT_HISTORICAL_INFO
