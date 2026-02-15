"""
Trust Registry
Maintains list of trusted sources (cameras, AI tools) and their certificates
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class TrustedSource:
    """Represents a trusted source (camera or AI tool)"""
    name: str
    type: str  # 'camera', 'ai_tool', 'software'
    certificate_hash: str
    public_key_hash: str
    verified: bool = True
    last_verified: str = ""
    notes: str = ""


class TrustRegistry:
    """Manages trusted sources and their certificates"""
    
    def __init__(self):
        self.trusted_sources: Dict[str, TrustedSource] = {}
        self._load_default_registry()
    
    def _load_default_registry(self):
        """Load default trusted sources"""
        
        # Trusted cameras
        trusted_cameras = [
            TrustedSource(
                name="Canon EOS R5",
                type="camera",
                certificate_hash="canon_eos_r5_cert_2024",
                public_key_hash="canon_eos_r5_key_2024",
                verified=True,
                notes="Professional DSLR with C2PA support"
            ),
            TrustedSource(
                name="Nikon Z9",
                type="camera",
                certificate_hash="nikon_z9_cert_2024",
                public_key_hash="nikon_z9_key_2024",
                verified=True,
                notes="Professional mirrorless camera"
            ),
            TrustedSource(
                name="iPhone 15 Pro",
                type="camera",
                certificate_hash="apple_iphone_15_cert_2024",
                public_key_hash="apple_iphone_15_key_2024",
                verified=True,
                notes="Apple smartphone with C2PA support"
            ),
        ]
        
        # Trusted AI tools
        trusted_ai_tools = [
            TrustedSource(
                name="DALL-E 3",
                type="ai_tool",
                certificate_hash="openai_dalle3_cert_2024",
                public_key_hash="openai_dalle3_key_2024",
                verified=True,
                notes="OpenAI image generation tool"
            ),
            TrustedSource(
                name="Midjourney",
                type="ai_tool",
                certificate_hash="midjourney_cert_2024",
                public_key_hash="midjourney_key_2024",
                verified=True,
                notes="Midjourney AI image generation"
            ),
            TrustedSource(
                name="Adobe Firefly",
                type="ai_tool",
                certificate_hash="adobe_firefly_cert_2024",
                public_key_hash="adobe_firefly_key_2024",
                verified=True,
                notes="Adobe's generative AI tool"
            ),
        ]
        
        for source in trusted_cameras + trusted_ai_tools:
            self.trusted_sources[source.name] = source
    
    def add_trusted_source(self, source: TrustedSource) -> bool:
        """Add a new trusted source"""
        if source.name in self.trusted_sources:
            print(f"[v0] Source {source.name} already exists")
            return False
        
        self.trusted_sources[source.name] = source
        print(f"[v0] Added trusted source: {source.name}")
        return True
    
    def verify_source(self, source_name: str, certificate_hash: str, 
                     public_key_hash: str) -> Tuple[bool, str]:
        """
        Verify if a source is trusted and its credentials match
        Returns: (is_valid, source_type)
        """
        
        if source_name not in self.trusted_sources:
            print(f"[v0] Source not in registry: {source_name}")
            return False, "UNKNOWN"
        
        source = self.trusted_sources[source_name]
        
        if not source.verified:
            print(f"[v0] Source is revoked: {source_name}")
            return False, source.type
        
        # Verify certificate and key hashes
        cert_match = source.certificate_hash == certificate_hash
        key_match = source.public_key_hash == public_key_hash
        
        if not (cert_match and key_match):
            print(f"[v0] Certificate/key hash mismatch for {source_name}")
            return False, source.type
        
        print(f"[v0] Source verification PASSED: {source_name} ({source.type})")
        return True, source.type
    
    def is_camera(self, source_name: str) -> bool:
        """Check if source is a camera"""
        if source_name in self.trusted_sources:
            return self.trusted_sources[source_name].type == "camera"
        return False
    
    def is_ai_tool(self, source_name: str) -> bool:
        """Check if source is an AI tool"""
        if source_name in self.trusted_sources:
            return self.trusted_sources[source_name].type == "ai_tool"
        return False
    
    def get_source_info(self, source_name: str) -> Optional[Dict]:
        """Get detailed information about a source"""
        if source_name in self.trusted_sources:
            return asdict(self.trusted_sources[source_name])
        return None
    
    def export_registry(self, filepath: str):
        """Export registry to JSON file"""
        try:
            data = {
                name: asdict(source) 
                for name, source in self.trusted_sources.items()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[v0] Registry exported to {filepath}")
        except Exception as e:
            print(f"[v0] Error exporting registry: {e}")
