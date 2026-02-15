#!/usr/bin/env python3
"""
Get trust registry as JSON
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trust_registry import TrustRegistry

if __name__ == '__main__':
    registry = TrustRegistry()
    
    # Convert registry to JSON-serializable format
    output = {}
    for name, source in registry.trusted_sources.items():
        output[name] = {
            'name': source.name,
            'type': source.type,
            'certificate_hash': source.certificate_hash,
            'public_key_hash': source.public_key_hash,
            'verified': source.verified,
            'last_verified': source.last_verified,
            'notes': source.notes,
        }
    
    print(json.dumps(output, indent=2))
