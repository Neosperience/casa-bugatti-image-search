import os
import logging

import streamlit as st

def load_env() -> None:
    logging.basicConfig(level=logging.INFO)