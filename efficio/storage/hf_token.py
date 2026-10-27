"""Find a Hugging Face token without it ever being typed into code.

Order: an explicit argument, the HF_TOKEN environment variable, Colab secrets, Kaggle
secrets (both named HF_TOKEN). Only where the token came from is ever logged, never the
token itself.
"""

import os

SECRET_NAME = "HF_TOKEN"
HOW_TO_ADD = (
    "Add a secret named HF_TOKEN holding a Hugging Face token with write access "
    "(huggingface.co/settings/tokens):\n"
    "  Colab: the key icon in the left sidebar, then turn on notebook access.\n"
    "  Kaggle: Add-ons > Secrets, then tick it for this notebook.\n"
    "  Elsewhere: set the HF_TOKEN environment variable.")


def _colab_secret():
    try:
        from google.colab import userdata
        return userdata.get(SECRET_NAME)
    except Exception:            # not on Colab, no such secret, or access not granted
        return None


def _kaggle_secret():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(SECRET_NAME)
    except Exception:            # not on Kaggle, or no such secret attached
        return None


def find_hf_token(token=None):
    """Returns (token or None, where it came from)."""
    if token:
        return token, "the token argument"
    if os.environ.get(SECRET_NAME):
        return os.environ[SECRET_NAME], "the HF_TOKEN environment variable"
    value = _colab_secret()
    if value:
        return value, "Colab secrets (HF_TOKEN)"
    value = _kaggle_secret()
    if value:
        return value, "Kaggle secrets (HF_TOKEN)"
    return None, "no token found (using a saved huggingface-cli login, if any)"
