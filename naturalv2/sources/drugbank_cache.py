"""DrugBank Cache Module.

This module provides a cache for DrugBank aliases, allowing for quick lookups of
drug names and their aliases.
"""

import ast
import gzip
import json
import os

import pandas as pd
from lxml import etree as ET  # noqa: N812


_aliases_cache: dict[int, list[str]] | None = None
_index_mapping: dict[str, int] | None = None


def get_drugbank_aliases(data_path: str, drug_name: str) -> list[str]:
    """Retrieve aliases for a given drug name from the DrugBank cache.

    This function loads the DrugBank data if not already loaded and returns
    a list of aliases for the specified drug name. The drug name is converted
    to lowercase for case-insensitive matching.

    Parameters
    ----------
    data_path : str
        The path to the directory containing DrugBank data files.
    drug_name : str
        The name of the drug for which aliases are to be retrieved.

    Returns
    -------
    list[str]
        A list of aliases for the specified drug name. If the drug name is not found,
        an empty list is returned.

    Raises
    -------
    FileNotFoundError
        If the DrugBank data files are not found in the specified path.
    RuntimeError
        If there is an error loading the DrugBank data.
    """
    _load_data(data_path)

    drug_index = _index_mapping.get(drug_name.lower())
    return _aliases_cache.get(drug_index, [])


def _load_data(data_path: str) -> None:
    """Load DrugBank aliases and indices from cached files or parse the XML file."""
    global _aliases_cache, _index_mapping  # noqa: PLW0603

    if _aliases_cache is not None:
        return

    alias_path = os.path.join(data_path, "drugbank_aliases.csv")
    index_path = os.path.join(data_path, "drugbank_indices.csv")
    try:
        if os.path.exists(alias_path) and os.path.exists(index_path):
            aliases_df = pd.read_csv(alias_path, index_col=0)
            with open(index_path, "r") as f:
                _index_mapping = json.load(f)
        else:
            file_path = os.path.join(data_path, "full_database.xml.gz")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"DrugBank data file not found: {file_path}")

            aliases_dicts, _index_mapping = _parse_drugbank_xml(file_path)
            aliases_df = pd.DataFrame(aliases_dicts)
            aliases_df.to_csv(alias_path)
            with open(index_path, "w") as f:
                json.dump(_index_mapping, f)
    except Exception as e:
        # Reset globals so retry is possible
        _aliases_cache = None
        _index_mapping = None
        raise RuntimeError(f"Failed to load DrugBank data: {e}") from e

    # Pre-process for faster lookups
    _aliases_cache = {}
    for _, row in aliases_df.iterrows():
        index = row["index"]
        if index not in _aliases_cache:
            _aliases_cache[index] = []
        _aliases_cache[index].extend(ast.literal_eval(row["alias_list"]))


def _parse_drugbank_xml(file_path: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Parse the DrugBank XML file and extract drug aliases and indices."""
    with gzip.open(file_path, "rt") as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        ns = "{http://www.drugbank.ca}"
        aliases_dicts = []
        index_mapping = {}

        # Helper to extract and append text if present
        def append_text(elem, aliases):
            if elem is not None and elem.text:
                aliases.append(elem.text.strip().lower())

        for index, drug in enumerate(root.findall(ns + "drug")):
            aliases: list[str] = []

            append_text(drug.find(ns + "name"), aliases)

            synonyms_elem = drug.find(ns + "synonyms")
            if synonyms_elem is not None:
                for syn_elem in synonyms_elem.findall(ns + "synonym"):
                    append_text(syn_elem, aliases)

            products_elem = drug.find(ns + "products")
            if products_elem is not None:
                for product_elem in products_elem.findall(ns + "product"):
                    append_text(product_elem.find(ns + "name"), aliases)

            intl_brands_elem = drug.find(ns + "international-brands")
            if intl_brands_elem is not None:
                for intl_brand_elem in intl_brands_elem.findall(
                    ns + "international-brand"
                ):
                    append_text(intl_brand_elem.find(ns + "name"), aliases)

            aliases = list(set(aliases))
            aliases_dicts.append({"index": index, "alias_list": str(aliases)})
            for alias in aliases:
                index_mapping[alias] = index

    return aliases_dicts, index_mapping
