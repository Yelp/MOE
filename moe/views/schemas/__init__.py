# -*- coding: utf-8 -*-
"""Request/Response schemas for the MOE REST interface and associated building blocks.

Contains:

    * :mod:`moe.views.schemas.bandit_pretty_view`: common schemas for the ``bandit_*`` endpoints
    * :mod:`moe.views.schemas.base_schemas`: basic building-block schemas for use in other, more complex schemas
    * :mod:`moe.views.schemas.gp_next_points_pretty_view`: common schemas for the ``gp_next_points_*`` endpoints
    * :mod:`moe.views.rest`: schemas for specific REST endpoints

.. Warning:: Outputs of colander schema serialization/deserialization should be treated as
  READ-ONLY. It appears that "missing=" and "default=" value are weak-copied (by reference).
  Thus changing missing/default fields in the output dict can modify the schema!

"""
