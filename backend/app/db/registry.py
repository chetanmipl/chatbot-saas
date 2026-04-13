# backend/app/db/registry.py
# This file's only job is to import all models so SQLAlchemy
# knows about them before create_all() runs.
# Nothing imports THIS file except main.py

from app.db.base import Base  # noqa
from app.models.tenant   import Tenant    # noqa
from app.models.user     import User      # noqa
from app.models.chatbot  import Chatbot   # noqa
from app.models.document import Document  # noqa
from app.models.message  import Message   # noqa