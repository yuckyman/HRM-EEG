"""
Test suite for hierarchical EEG processing.
"""

from .test_attention import test_temporal_attention, test_fast_rnn, test_slow_rnn, test_integration
from .test_organization import test_directory_structure, test_file_paths, test_data_migration, test_config_import

__all__ = [
    'test_temporal_attention',
    'test_fast_rnn',
    'test_slow_rnn', 
    'test_integration',
    'test_directory_structure',
    'test_file_paths',
    'test_data_migration',
    'test_config_import'
] 