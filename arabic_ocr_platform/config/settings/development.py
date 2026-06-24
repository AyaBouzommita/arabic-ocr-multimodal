from .base import *
from .base import BASE_DIR

DEBUG = True

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Celery local / eager mode (can be run synchronously in development without Redis if desired)
# CELERY_TASK_ALWAYS_EAGER = True
