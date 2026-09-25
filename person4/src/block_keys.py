"""
block_keys.py

Generate multiple blocking keys for candidate generation.
"""

from similarity import (
    normalize_business_name,
    normalize_address,
    first_token,
    first_two_tokens,
    first_three_char,
    address_prefix,
)


def generate_name_keys(name: str):
    """
    Generate multiple business-name blocking keys.
    """

    name = normalize_business_name(name)

    keys = set()

    if not name:
        return keys

    keys.add(first_token(name))
    keys.add(first_two_tokens(name))
    keys.add(first_three_char(name))

    return keys


def generate_address_keys(address: str):
    """
    Generate multiple address blocking keys.
    """

    address = normalize_address(address)

    keys = set()

    if not address:
        return keys

    keys.add(address_prefix(address))

    return keys


def generate_all_keys(name, address):

    return generate_name_keys(name).union(
        generate_address_keys(address)
    )