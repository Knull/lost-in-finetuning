"""Lost in Fine-Tuning: retain, repair or reject a Hindi adapter of Tiny Aya Global.

Core modules (config, io, schemas, cachekey, splits, selection, stats, judge.labels) are pure
Python and tested offline. Everything that touches a model lives behind the ``gpu`` extra.
"""

__version__ = "0.1.0"
