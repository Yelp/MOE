# -*- coding: utf-8 -*-
"""Request/response schemas for the MOE REST interface.

Contains schemas for each endpoint represented in :mod:`moe.views.rest`.

.. Warning:: Outputs of colander schema serialization/deserialization should be treated as
  READ-ONLY. It appears that "missing=" and "default=" value are weak-copied (by reference).
  Thus changing missing/default fields in the output dict can modify the schema!

"""
