#!/usr/bin/env python
"""Test script to verify imports without running full Streamlit app."""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    print("Testing imports...")
    
    print("1. Importing streamlit...")
    import streamlit as st
    print("   OK")
    
    print("2. Importing pages.qa_demo.main_dev...")
    from pages.qa_demo import main_dev
    print("   OK")
    
    print("3. Importing pages.detect_demo.main_dev...")
    from pages.detect_demo import main_dev as detect_demo
    print("   OK")
    
    print("4. Importing pages.doc_parse_demo.main_dev...")
    from pages.doc_parse_demo import main_dev as doc_parse_demo
    print("   OK")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"\n❌ Import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
